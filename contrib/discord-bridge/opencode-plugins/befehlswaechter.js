/**
 * Befehlswächter — prüft Bash-Befehle, bevor opencode sie ausführt.
 *
 * Vier Stufen, damit der Normalfall ohne Verzögerung durchläuft:
 *   1. harte Sperre  — offensichtlich zerstörerische Muster, sofort blockiert
 *   2. Fremdpfade    — Zugriff außerhalb des Projekts wird nicht mehr sofort
 *                      abgelehnt, sondern über die Discord-Bridge von Hermes
 *                      einmalig zur Freigabe vorgelegt (siehe unten)
 *   3. Freiliste     — bekannte harmlose Befehle, sofort durchgelassen
 *   4. Modellprüfung — alles Übrige geht an ein schnelles Modell (Claude Haiku)
 *
 * Gedacht als Netz für den Betrieb mit `--auto`. Ersetzt keine Sorgfalt:
 * ein Sprachmodell ist eine zweite Meinung, keine Sicherheitsgarantie.
 *
 * Rückfrage über Discord (Stufe 2)
 * --------------------------------
 * opencodes eigenes Permission-System taugt dafür nicht: im Automatikmodus
 * beantwortet die TUI jede native Anfrage sofort mit „once“, bevor Discord
 * sie sieht. Deshalb hält dieses Plugin den Werkzeugaufruf selbst an und
 * spricht mit der Bridge über einen privaten Ablageordner
 * (`$HERMES_HOME/opencode_bridge`, sonst `~/.hermes/opencode_bridge`):
 *
 *   requests/<id>.json   — schreibt dieses Plugin (atomar per rename)
 *   decisions/<id>.json  — schreibt die Bridge: {"decision": "once"|"reject"}
 *
 * Die Bridge zeigt Befehl, Pfad, Projektordner und Zugriffsart in Discord
 * und bietet nur „einmal erlauben“ oder „ablehnen“ an. Blockiert bleibt der
 * Zugriff bei Ablehnung, Zeitablauf, nicht laufendem Gateway und bei jeder
 * unklaren Antwort. Dauerhafte Freigaben gibt es nicht — jede Anfrage gilt
 * für genau einen Befehl.
 */

import crypto from "node:crypto"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

// Gemessen an einem Testbefehl: Haiku 0,9 s und korrektes Urteil.
// GLM-4.7-Flash war zwar kostenlos und mit 0,2 s schneller, lief aber
// reproduzierbar in HTTP 429 — ein Wächter darf unter Last nicht ausfallen.
const MODELL = "claude-haiku-4.5"
const ENDPUNKT = "https://api.githubcopilot.com/chat/completions"

// Ablage für die Discord-Rückfrage
const HERMES_HOME = process.env.HERMES_HOME || path.join(os.homedir(), ".hermes")
const ABLAGE = path.join(HERMES_HOME, "opencode_bridge")
const ANFRAGEN = path.join(ABLAGE, "requests")
const ANTWORTEN = path.join(ABLAGE, "decisions")
// Die Bridge lehnt nach ihrem eigenen Timeout (Standard 300 s) ab; das
// Plugin wartet etwas länger, damit diese Absage normalerweise zuerst kommt.
const WARTEZEIT_S = Number(process.env.BEFEHLSWAECHTER_WARTEZEIT_S) || 330
const ABFRAGE_MS = 500
// Ist der Heartbeat des Gateways älter, gilt die Bridge als ausgefallen —
// dann wird sofort blockiert statt minutenlang zu warten.
const HEARTBEAT_MAX_ALTER_S = 120

// 1. Immer blockieren — unabhängig davon, was das Modell meint
const GESPERRT = [
  /\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]/, // rm -rf und Varianten
  /\bsudo\b/,
  /\bmkfs|\bdd\s+if=|\bdiskutil\s+(erase|partition)/,
  /:\(\)\s*\{.*\}\s*;\s*:/, // Fork-Bombe
  /\bcurl\b[^|]*\|\s*(ba)?sh/, // curl … | sh
  /\bwget\b[^|]*\|\s*(ba)?sh/,
  /\bgit\s+push\b.*(--force|-f)\b/,
  /\bgit\s+reset\s+--hard\b/,
  /\bchmod\s+-R\s+777/,
  />\s*\/dev\/(disk|rdisk)/,
  /\bkillall\b|\bshutdown\b|\breboot\b/,
  // Zugangsdaten — egal mit welchem Befehl
  /\.ssh\/|\.aws\/|\.gnupg\/|\.netrc\b/,
  /\.env\b|auth\.json|credentials|id_[re]d?sa\b|\.pem\b|\.p12\b/,
  /\bsecurity\s+(find|dump)-(generic|internet)-password|\blogin\.keychain/,
  /\bhistory\b.*\|/,
]

// 3. Ohne Rückfrage erlauben — der Alltag dieses Projekts.
//    Lesebefehle nur auf projektrelative Pfade: kein ~, kein /, kein ..
const IM_PROJEKT = "(?![^\\n]*(~|\\s\\/|\\.\\.\\/|\\$HOME))"
const FREI = [
  new RegExp(`^typst\\s${IM_PROJEKT}`),
  new RegExp(`^typstyle\\s${IM_PROJEKT}`),
  new RegExp(`^(pdftoppm|pdfinfo)\\s${IM_PROJEKT}`),
  new RegExp(`^(ls|cat|head|tail|wc|file|grep|rg)\\s${IM_PROJEKT}`),
  new RegExp(`^mkdir\\s+-p\\s${IM_PROJEKT}`),
  new RegExp(`^(python3?|\\.)\\s*\\/?tools\\/${IM_PROJEKT}`),
  /^ls$/, /^pwd$/,
  /^(which|basename|dirname|echo)\s/,
  /^git\s+(status|diff|log|show|add|branch)\b/,
]

function schluessel() {
  try {
    const p = path.join(os.homedir(), ".local/share/opencode/auth.json")
    return JSON.parse(fs.readFileSync(p, "utf8"))["github-copilot"]?.access
  } catch {
    return null
  }
}

async function modellUrteil(befehl, verzeichnis, key) {
  const anweisung =
    `Du prüfst einen Shell-Befehl, den ein KI-Agent ausführen will.\n` +
    `Arbeitsverzeichnis: ${verzeichnis}\n` +
    `Befehl: ${befehl}\n\n` +
    `Blockiere, wenn der Befehl: Daten außerhalb des Arbeitsverzeichnisses löscht ` +
    `oder verändert; Systemeinstellungen ändert; Dinge aus dem Netz herunterlädt ` +
    `und ausführt; Zugangsdaten ausliest oder verschickt; Prozesse beendet, die ` +
    `nicht zum Projekt gehören.\n\n` +
    `Antworte mit genau einem Wort: OK oder STOP. ` +
    `Bei STOP hänge nach einem Doppelpunkt eine kurze Begründung an.`

  // Zeitgrenze: der Wächter darf die Arbeit nicht anhalten
  const abbruch = AbortSignal.timeout(8000)
  const antwort = await fetch(ENDPUNKT, {
    method: "POST",
    signal: abbruch,
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
      "Copilot-Integration-Id": "vscode-chat",
      "Editor-Version": "vscode/1.95.0",
    },
    body: JSON.stringify({
      model: MODELL,
      max_tokens: 80,
      temperature: 0,
      messages: [{ role: "user", content: anweisung }],
    }),
  })
  if (!antwort.ok) return { ok: true, grund: `Prüfung übersprungen (HTTP ${antwort.status})` }
  const d = await antwort.json()
  const text = (d.choices?.[0]?.message?.content ?? "").trim()
  if (/^STOP/i.test(text)) return { ok: false, grund: text.replace(/^STOP:?\s*/i, "") }
  return { ok: true }
}

// ---------------------------------------------------------------------------
// Fremdpfade erkennen
// ---------------------------------------------------------------------------

/** Zerlegt einen Befehl in Wörter; Anführungszeichen bleiben zusammen. */
function woerter(befehl) {
  return (befehl.match(/"[^"]*"|'[^']*'|\S+/g) || []).map((w) => w.replace(/^["']|["']$/g, ""))
}

/** Löst ~, $HOME und relative Angaben zu einem absoluten Pfad auf. */
function absolut(wort, projekt) {
  const p = wort.replace(/^~(?=\/|$)/, os.homedir()).replace(/^\$HOME(?=\/|$)/, os.homedir())
  return path.resolve(projekt, p)
}

/**
 * Welche Pfade im Befehl liegen außerhalb des Projekts?
 * Deterministische Regel — hier wird dem Modell bewusst nicht vertraut:
 * im Test hat es `grep -r password ~/` durchgewinkt.
 * Geprüft werden nur Wörter, die als Pfad BEGINNEN (auch nach `flag=`) —
 * sonst schlägt "Zuordnungen/AO.typ" fälschlich an.
 */
export function ausserhalb(befehl, projekt) {
  const gefunden = []
  for (const roh of woerter(befehl)) {
    const wort = roh.replace(/^[\w.-]+=(?=~\/|\/|\$HOME|\.\.)/, "")
    if (!/^(~(\/|$)|\$HOME(\/|$)|\/|\.\.(\/|$))/.test(wort)) continue
    const pfad = absolut(wort, projekt)
    if (pfad === projekt || pfad.startsWith(projekt + path.sep)) continue
    if (/^\/(usr|bin|sbin|opt|Applications)\//.test(pfad)) continue // Programmaufruf
    if (/^\/dev\/(null|stdout|stderr)$/.test(pfad)) continue
    if (!gefunden.includes(pfad)) gefunden.push(pfad)
  }
  return gefunden
}

const LESEND = new Set([
  "cat", "head", "tail", "less", "more", "wc", "file", "stat", "grep", "rg", "egrep",
  "fgrep", "find", "fd", "ls", "tree", "du", "df", "diff", "cmp", "md5", "shasum",
  "sha256sum", "pdfinfo", "pdftotext", "pdftoppm", "identify", "exiftool", "realpath",
  "readlink", "basename", "dirname", "test", "[", "strings", "xxd", "hexdump", "sort",
  "uniq", "cut", "awk", "jq", "yq", "open", "qlmanage", "mdls", "xattr", "lsof",
])
const SCHREIBEND = new Set([
  "cp", "mv", "rm", "rmdir", "mkdir", "touch", "ln", "chmod", "chown", "chgrp", "tee",
  "truncate", "rsync", "install", "unzip", "zip", "tar", "dd", "patch", "ditto",
])

/** Teilt an && || ; | außerhalb von Anführungszeichen. */
function abschnitte(befehl) {
  const teile = []
  let aktuell = ""
  let quote = null
  for (let i = 0; i < befehl.length; i++) {
    const c = befehl[i]
    if (quote) {
      aktuell += c
      if (c === quote) quote = null
      continue
    }
    if (c === '"' || c === "'") { quote = c; aktuell += c; continue }
    if (c === "&" && befehl[i + 1] === "&") { teile.push(aktuell); aktuell = ""; i++; continue }
    if (c === "|" && befehl[i + 1] === "|") { teile.push(aktuell); aktuell = ""; i++; continue }
    if (c === ";" || c === "|") { teile.push(aktuell); aktuell = ""; continue }
    aktuell += c
  }
  teile.push(aktuell)
  return teile.map((t) => t.trim()).filter(Boolean)
}

/**
 * Art des Zugriffs auf die Fremdpfade: lesen, schreiben, ausführen oder
 * unklar. Nur eine Einordnung für den Menschen — entschieden wird in Discord.
 */
export function zugriffsart(befehl, projekt) {
  const arten = new Set()
  for (const teil of abschnitte(befehl)) {
    const fremd = ausserhalb(teil, projekt)
    if (!fremd.length) continue
    const w = woerter(teil).filter((x) => !/^[A-Za-z_][A-Za-z0-9_]*=/.test(x)) // VAR=… überspringen
    const programm = path.basename(w[0] ?? "")
    // Umleitung in einen Fremdpfad ist immer ein Schreibzugriff
    const umleitung = teil.match(/>>?\s*("[^"]*"|'[^']*'|\S+)/)
    if (umleitung && ausserhalb(umleitung[1].replace(/^["']|["']$/g, ""), projekt).length) {
      arten.add("schreiben")
      continue
    }
    if (programm === "sed") { arten.add(/\s-[a-zA-Z]*i/.test(teil) ? "schreiben" : "lesen"); continue }
    if (fremd.some((p) => absolut(w[0] ?? "", projekt) === p)) { arten.add("ausführen"); continue }
    if (SCHREIBEND.has(programm)) arten.add("schreiben")
    else if (LESEND.has(programm)) arten.add("lesen")
    else if (programm === "cd" || programm === "pushd") arten.add("ausführen")
    else arten.add("ausführen")
  }
  if (arten.has("schreiben")) return "schreiben"
  if (arten.has("ausführen")) return "ausführen"
  if (arten.has("lesen")) return "lesen"
  return "unklar"
}

// ---------------------------------------------------------------------------
// Rückfrage über die Discord-Bridge
// ---------------------------------------------------------------------------

/** Läuft das Hermes-Gateway mit verbundenem Discord? Sonst Grund zurückgeben. */
export function brueckeAusgefallen(hermesHome = HERMES_HOME) {
  let heartbeat
  try {
    heartbeat = JSON.parse(fs.readFileSync(path.join(hermesHome, "state", "gateway.heartbeat"), "utf8"))
  } catch {
    return "kein Gateway-Heartbeat gefunden"
  }
  const alter = (Date.now() - Date.parse(heartbeat.updated_at ?? "")) / 1000
  if (!(alter < HEARTBEAT_MAX_ALTER_S)) return `Gateway-Heartbeat ist ${Math.round(alter) || "?"} s alt`
  let zustand
  try {
    zustand = JSON.parse(fs.readFileSync(path.join(hermesHome, "gateway_state.json"), "utf8"))
  } catch {
    return "gateway_state.json nicht lesbar"
  }
  if (zustand.gateway_state !== "running") return `Gateway-Zustand: ${zustand.gateway_state ?? "unbekannt"}`
  const discord = zustand.platforms?.discord?.state
  if (discord !== "connected") return `Discord ist ${discord ?? "nicht konfiguriert"}`
  return null
}

function schlafen(ms) {
  return new Promise((r) => setTimeout(r, ms))
}

function leiseLoeschen(p) {
  try { fs.unlinkSync(p) } catch {}
}

/**
 * Legt die Anfrage ab und wartet auf die Entscheidung der Bridge.
 * Rückgabe: { erlaubt: boolean, grund: string }. Alles außer einer sauberen
 * „once“-Antwort mit passender ID gilt als Ablehnung.
 */
export async function brueckeFragen(anfrage, optionen = {}) {
  // Gegen den Plugin-Fabrik-Aufruf beim Laden (Kontextobjekt statt Anfrage).
  if (!anfrage || !Array.isArray(anfrage.pfade)) return { erlaubt: false, grund: "ungültige Anfrage" }
  const ablage = optionen.ablage ?? ABLAGE
  const anfragen = path.join(ablage, "requests")
  const antworten = path.join(ablage, "decisions")
  const wartezeitS = optionen.wartezeitS ?? WARTEZEIT_S
  const abfrageMs = optionen.abfrageMs ?? ABFRAGE_MS

  const ausfall = optionen.ausfallpruefung === false ? null : brueckeAusgefallen(optionen.hermesHome)
  if (ausfall) return { erlaubt: false, grund: `Discord-Bridge nicht erreichbar: ${ausfall}` }

  const id = crypto.randomUUID()
  const jetzt = Date.now() / 1000
  const nutzlast = {
    version: 1,
    id,
    agent: "opencode",
    created_at: jetzt,
    expires_at: jetzt + wartezeitS,
    session_id: anfrage.sessionID ?? "",
    project: anfrage.projekt,
    command: anfrage.befehl,
    path: anfrage.pfade.join(", "),
    access: anfrage.zugriff,
  }
  const anfrageDatei = path.join(anfragen, `${id}.json`)
  const antwortDatei = path.join(antworten, `${id}.json`)
  try {
    for (const d of [ablage, anfragen, antworten]) fs.mkdirSync(d, { recursive: true, mode: 0o700 })
    const tmp = path.join(anfragen, `.tmp-${id}`)
    fs.writeFileSync(tmp, JSON.stringify(nutzlast), { mode: 0o600 })
    fs.renameSync(tmp, anfrageDatei)
  } catch (e) {
    return { erlaubt: false, grund: `Anfrage konnte nicht abgelegt werden (${e.message})` }
  }

  const ende = Date.now() + wartezeitS * 1000
  try {
    while (Date.now() < ende) {
      let roh
      try {
        roh = fs.readFileSync(antwortDatei, "utf8")
      } catch {
        await schlafen(abfrageMs)
        continue
      }
      let antwort
      try {
        antwort = JSON.parse(roh)
      } catch {
        return { erlaubt: false, grund: "unlesbare Antwort der Bridge" }
      }
      if (antwort?.version !== 1 || antwort?.id !== id) {
        return { erlaubt: false, grund: "Antwort der Bridge passt nicht zur Anfrage" }
      }
      if (antwort.decision === "once") return { erlaubt: true, grund: "in Discord einmalig erlaubt" }
      if (antwort.decision === "reject") {
        return {
          erlaubt: false,
          grund: antwort.source === "timeout" ? "in Discord nicht rechtzeitig beantwortet" : "in Discord abgelehnt",
        }
      }
      return { erlaubt: false, grund: `unbekannte Entscheidung „${String(antwort.decision)}“` }
    }
    return { erlaubt: false, grund: `keine Antwort innerhalb von ${wartezeitS} s` }
  } finally {
    leiseLoeschen(anfrageDatei)
    leiseLoeschen(antwortDatei)
  }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

export const Befehlswaechter = async ({ directory }) => {
  const key = schluessel()
  const projekt = path.resolve(directory ?? process.cwd())
  return {
    "tool.execute.before": async (input, output) => {
      if (input.tool !== "bash") return
      const befehl = String(output.args?.command ?? "").trim()
      if (!befehl) return

      for (const muster of GESPERRT) {
        if (muster.test(befehl)) {
          throw new Error(
            `Befehlswächter: blockiert (harte Sperre).\nBefehl: ${befehl}\n` +
              `Falls beabsichtigt, führe ihn selbst im Terminal aus.`,
          )
        }
      }

      const fremd = ausserhalb(befehl, projekt)
      if (fremd.length) {
        const zugriff = zugriffsart(befehl, projekt)
        const ergebnis = await brueckeFragen({
          befehl,
          pfade: fremd,
          projekt,
          zugriff,
          sessionID: input.sessionID,
        })
        if (ergebnis.erlaubt) return // einmalige Freigabe — keine weitere Prüfung
        throw new Error(
          `Befehlswächter: Zugriff außerhalb des Projekts blockiert (${ergebnis.grund}).\n` +
            `Befehl: ${befehl}\nPfad: ${fremd.join(", ")}\nProjekt: ${projekt}\nZugriff: ${zugriff}\n` +
            `Nicht erneut versuchen. Falls beabsichtigt, führe ihn selbst im Terminal aus.`,
        )
      }

      if (FREI.some((m) => m.test(befehl))) return
      if (!key) return // ohne Schlüssel keine Modellprüfung, Rest regelt die Berechtigung

      try {
        const urteil = await modellUrteil(befehl, directory, key)
        if (!urteil.ok) {
          throw new Error(
            `Befehlswächter: ${MODELL} rät ab.\nBefehl: ${befehl}\nGrund: ${urteil.grund}`,
          )
        }
      } catch (e) {
        if (String(e.message).startsWith("Befehlswächter:")) throw e
        // Netzfehler dürfen die Arbeit nicht anhalten
      }
    },
  }
}
