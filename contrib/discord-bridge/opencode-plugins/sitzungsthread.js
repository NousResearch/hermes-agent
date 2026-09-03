/**
 * Sitzungsthread — meldet jede opencode-Sitzung an die Hermes-Discord-Bridge,
 * die daraus einen Thread im Bridge-Kanal macht:
 *
 *   Sitzungsstart  → Thread mit Startzeit, Projektordner und Prompt
 *   weitere Prompts → „Neuer Prompt“ im Thread
 *   Ende jeder Runde → „Antwort“ (letzter Assistententext) im Thread
 *   Unteragenten   → ihre Rückfragen landen im Thread der Elternsitzung
 *
 * Die Rückfragen des Befehlswächters derselben Sitzung stellt die Bridge
 * ebenfalls in diesem Thread. Transport ist dieselbe Ablage wie beim
 * Befehlswächter (`$HERMES_HOME/opencode_bridge/requests`), nur ohne
 * Antwortdatei: Meldungen werden abgelegt und nicht abgewartet. Ist die
 * Bridge aus, bleiben die Dateien liegen, bis sie verfallen — die Arbeit
 * von opencode hält das nie an.
 */

import crypto from "node:crypto"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

const HERMES_HOME = process.env.HERMES_HOME || path.join(os.homedir(), ".hermes")
const ABLAGE = path.join(HERMES_HOME, "opencode_bridge")
const GUELTIG_S = 3600 // die Bridge verwirft ältere Meldungen
const MAX_ZEICHEN = 1700
const GUELTIGE_MELDUNGEN = new Set(["start", "prompt", "result", "child"])

function kuerzen(text, max = MAX_ZEICHEN) {
  const t = String(text || "").trim()
  return t.length <= max ? t : t.slice(0, max - 1).trimEnd() + "…"
}

/** Legt eine Meldung atomar in der Ablage ab; Fehler werden geschluckt. */
export function melden(meldung, ablage = ABLAGE) {
  // opencode ruft JEDE Export-Funktion beim Laden als Plugin-Fabrik mit dem
  // Kontext {client, project, directory, ...} auf. Ohne echtes notice-Feld
  // ist das kein Meldeaufruf — nichts ablegen.
  if (!meldung || typeof meldung !== "object" || !GUELTIGE_MELDUNGEN.has(meldung.notice)) return false
  const anfragen = path.join(ablage, "requests")
  const id = crypto.randomUUID()
  const jetzt = Date.now() / 1000
  const nutzlast = {
    version: 1,
    id,
    agent: "opencode",
    kind: "notice",
    created_at: jetzt,
    expires_at: jetzt + GUELTIG_S,
    started_at: jetzt,
    ...meldung,
    text: kuerzen(meldung.text),
  }
  try {
    fs.mkdirSync(anfragen, { recursive: true, mode: 0o700 })
    const tmp = path.join(anfragen, `.tmp-${id}`)
    fs.writeFileSync(tmp, JSON.stringify(nutzlast), { mode: 0o600 })
    fs.renameSync(tmp, path.join(anfragen, `${id}.json`))
    return true
  } catch {
    return false
  }
}

/**
 * Verfolgt Sitzungen und Nachrichten aus dem Ereignisstrom und liefert die
 * Meldungen, die abzulegen sind. Reine Zustandslogik, testbar ohne Discord.
 */
export function sitzungsZustand(projekt) {
  // Reale opencode-Ereignisse (run- wie TUI-Modus, verifiziert):
  //  - Der Nutzer-Textteil kommt OHNE time.end (der ganze Prompt steht sofort
  //    im part.text). Deshalb wird hier NICHT auf end gewartet.
  //  - Rolle (message.updated) und Textteil (message.part.updated) können in
  //    beliebiger Reihenfolge eintreffen; darum wird beides gepuffert und die
  //    Meldung erzeugt, sobald Rolle UND Text bekannt sind.
  //  - Assistenten-Text streamt als Schnappschuss; der letzte Stand pro
  //    Nachricht ist die fertige Antwort und wird beim session.idle gemeldet.
  const rollen = new Map() // messageID → role
  const texte = new Map() // messageID → { sid, text }
  const sitzungen = new Map() // sessionID → Zustand
  const eltern = new Map() // child sessionID → parent

  function eintrag(sid) {
    if (!sitzungen.has(sid)) {
      sitzungen.set(sid, { gestartet: false, gemeldetePrompts: new Set(), letzteAntwort: null, gemeldeteAntwort: null })
    }
    return sitzungen.get(sid)
  }

  /** Erzeugt die Meldung für eine Nachricht, sobald Rolle und Text feststehen. */
  function auswerten(mid) {
    const rolle = rollen.get(mid)
    const eintragText = texte.get(mid)
    if (!rolle || !eintragText) return []
    const { sid, text } = eintragText
    const s = eintrag(sid)
    if (rolle === "user") {
      if (eltern.has(sid)) return [] // Unteragenten-Prompts nicht melden
      if (s.gemeldetePrompts.has(mid)) return []
      s.gemeldetePrompts.add(mid)
      if (!s.gestartet) {
        s.gestartet = true
        return [{ notice: "start", session_id: sid, project: projekt, text }]
      }
      return [{ notice: "prompt", session_id: sid, project: projekt, text }]
    }
    if (rolle === "assistant") s.letzteAntwort = { id: mid, text }
    return []
  }

  function verarbeite(ev) {
    const p = ev?.properties ?? {}
    switch (ev?.type) {
      case "session.created":
      case "session.updated": {
        const info = p.info ?? {}
        if (info.id && info.parentID && !eltern.has(info.id)) {
          eltern.set(info.id, info.parentID)
          return [{ notice: "child", session_id: info.id, parent_session_id: info.parentID, project: projekt }]
        }
        return []
      }
      case "message.updated": {
        const info = p.info ?? {}
        if (info.id && info.role) {
          rollen.set(info.id, info.role)
          return auswerten(info.id)
        }
        return []
      }
      case "message.part.updated": {
        const part = p.part ?? {}
        if (part.type !== "text" || !part.sessionID || !part.messageID) return []
        texte.set(part.messageID, { sid: part.sessionID, text: part.text ?? "" })
        return auswerten(part.messageID)
      }
      case "session.idle":
      case "session.status": {
        if (ev.type === "session.status" && p.status?.type !== "idle") return []
        const sid = p.sessionID
        if (!sid || eltern.has(sid)) return []
        const s = sitzungen.get(sid)
        if (!s?.letzteAntwort || s.gemeldeteAntwort === s.letzteAntwort.id) return []
        s.gemeldeteAntwort = s.letzteAntwort.id
        return [{ notice: "result", session_id: sid, project: projekt, text: s.letzteAntwort.text }]
      }
      default:
        return []
    }
  }
  return { verarbeite }
}

export const Sitzungsthread = async ({ directory }) => {
  const projekt = path.resolve(directory ?? process.cwd())
  const zustand = sitzungsZustand(projekt)
  return {
    event: async ({ event }) => {
      for (const meldung of zustand.verarbeite(event)) melden(meldung)
    },
  }
}
