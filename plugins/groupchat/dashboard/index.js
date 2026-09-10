/* Groupchat dashboard extension. Uses the host's React and authenticated SDK. */
(() => {
  const sdk = window.__HERMES_PLUGIN_SDK__;
  const { React, api, fetchJSON } = sdk;
  const { useState, useEffect } = sdk.hooks;
  const h = React.createElement;
  const providerChoices = [["mistral", "mistral"], ["openrouter", "OpenRouter"], ["openai-codex", "Codex (OAuth)"]];
  const endpoint = profile => "/api/plugins/groupchat/settings?profile=" + encodeURIComponent(profile);

  // Subscribe to the host language selector; older hosts and untranslated
  // locales use English. No separate plugin language preference is stored.
  const useI18n = sdk.useI18n || (() => ({ locale: "en" }));
  const messages = {
  "en": {
    "languages": "Defaults include German and English acknowledgements. Add other languages as additional lines. Existing custom lists are preserved; use Restore defaults to adopt updated defaults.",
    "logs": "Decision logs",
    "logsHelp": "Profile-local JSONL files, created when a decision is recorded. Entries include UTC time, room/event identifiers, decision and reason code; pattern matches also include the rule index and SHA-256 fingerprint. Model decisions include provider/model and fallback information where available. Message bodies, prompts, credentials and free-form model rationales are not logged.",
    "logsRetention": "Each file rotates at 10 MiB to .jsonl.1 (one previous file retained; owner-only permissions). Outbound send means allowed by the filter, not confirmed delivery. Logs remain on the gateway server, not this browser’s computer.",
    "logDirectory": "Log directory",
    "logDirectoryHelp": "For the selected profile. Filenames identify the channel and filter; files are created when decisions are recorded.",
    "contextDirectory": "Persistent score-1 context directory",
    "contextDirectoryHelp": "Restart-durable private state is stored as passive-context.<channel>.json with owner-only permissions. It contains retained message text and is removed after successful delivery to the agent.",
    "attribution": "Groupchat coordination was developed by RechnerLotsen.",
    "patterns": "Message patterns",
    "patternHelp": "One Python regular expression per line. Changes replace the defaults. Use ^...$ for a whole-message match. Empty lists disable these patterns, not AI checks or lifecycle handling. Only use trusted patterns; complex expressions can slow message processing.",
    "inboundPatterns": "System-message patterns",
    "inboundPatternHelp": "Case-sensitive matching at the start of the first line, before mention routing. Prefix with (?i) to ignore case.",
    "outboundPatterns": "Suppressed-reply patterns",
    "outboundPatternHelp": "Case-insensitive search in the raw and Markdown-stripped reply, before the length threshold and AI check.",
    "multilinePatterns": "Additional multiline progress patterns",
    "multilineHelp": "The first line and at least one more line must match a system or multiline pattern. These patterns alone do not suppress single-line messages.",
    "interruptNotice": "Literal text — blacklisted phrases",
    "interruptHelp": "One literal phrase per line (not regex). Discard only without an explicit mention and when occurrences of one listed phrase cover more than 50% of the message. Exactly 50% passes. Case-sensitive; whitespace is normalized. Different phrases are not added together. Empty list disables this check. Lines starting with # are comments.",
    "comments": "Lines starting with # are comments, e.g. # English keywords. Comments are saved but never matched. In regex rules, use \\# to match a leading # literally.",
    "restorePatterns": "Restore defaults",
    "direct": "Mistral direct",
    "saved": "Saved. Restart this profile’s gateway to apply the changes.",
    "primary": "Primary",
    "provider": "Provider",
    "model": "Model",
    "profileModel": "Leave blank to use the profile’s Codex model",
    "modelId": "Model ID",
    "eyebrow": "HERMES · COLLABORATION",
    "subtitle": "Recognize relevant messages. Coordinate replies. Prevent bot pingpong.",
    "profile": "Groupchat profile",
    "discardConfirm": "Discard unsaved Groupchat changes?",
    "dashboardProfile": "Dashboard profile",
    "loadError": "Could not load settings.",
    "loading": "Loading settings…",
    "enable": "Enable Groupchat",
    "enableHint": "One addon for inbound relevance and outbound pingpong protection.",
    "permissions": "Existing room, user and bot permissions still apply. Groupchat does not grant additional access.",
    "mentionRequirement": "For relevance-based participation in unmentioned group messages, Matrix must use require_mention=false. With true, Hermes receives only direct mentions; the outbound pingpong guard still works.",
    "participation": "Matrix participation",
    "participationHelp": "Saved effective settings for all running named gateways. Active means Groupchat and its relevance filter are enabled for Matrix and require_mention is false. Changes require a gateway restart.",
    "activeParticipant": "active",
    "mentionsOnly": "mentions only",
    "groupchatOff": "Groupchat off",
    "filterAI": "Filter AI",
    "modelHint": "Both filters use this model selection. Fallbacks are tried in order. Credentials stay in the respective Hermes profile.",
    "addFallback": "+ Add fallback",
    "inbound": "INBOUND",
    "relevance": "Relevance filter",
    "scoreMessages": "Score messages",
    "relevanceHint": "Direct mentions are passed through immediately. Other messages are buffered by relevance; process updates are filtered out.",
    "delays": "Delay by relevance (seconds)",
    "scoreSemantics": "Score 0 is discarded. Score 1 remains passive context without a timer. Only scores 2–5 can start an agent turn; the filter always scores the newest message, using recent messages only as context.",
    "infoDelay": "Information-only delay (seconds)",
    "maxBuffer": "Maximum buffered messages",
    "outbound": "OUTBOUND",
    "guard": "Pingpong guard",
    "filterReplies": "Filter replies",
    "guardHint": "Known filler replies and silence markers are suppressed. The shared filter AI checks short, ambiguous replies against the conversation context.",
    "minChars": "Skip AI checking at this character count",
    "failOpen": "Replies are sent if the outbound check fails. On every selected channel, replies are buffered before publication when the guard is enabled.",
    "unsaved": "Unsaved changes",
    "loaded": "Settings loaded · Changes require a gateway restart",
    "discard": "Discard",
    "saving": "Saving…",
    "save": "Save",
    "fallback": "Fallback {number}",
    "removeFallback": "Remove fallback {number}",
    "score": "Score {number}"
  },
  "de": {
    "languages": "Die Standards enthalten deutsche und englische Bestätigungen. Weitere Sprachen können als zusätzliche Zeilen ergänzt werden. Eigene Listen bleiben erhalten; aktualisierte Standards lassen sich mit Standards wiederherstellen übernehmen.",
    "logs": "Entscheidungslogs",
    "logsHelp": "Profilbezogene JSONL-Dateien, die beim ersten Eintrag entstehen. Enthalten UTC-Zeit, Raum-/Ereigniskennungen, Entscheidung und Grundcode; bei Mustertreffern auch Regelnummer und SHA-256-Fingerabdruck. KI-Entscheidungen enthalten verfügbare Provider-/Modell- und Fallback-Angaben. Keine Nachrichtentexte, Prompts, Zugangsdaten oder frei formulierten KI-Begründungen.",
    "logsRetention": "Rotation je Datei bei 10 MiB nach .jsonl.1 (eine Vorgängerdatei; nur für den Eigentümer zugänglich). Ausgehend bedeutet send: vom Filter erlaubt, nicht erfolgreich zugestellt. Die Dateien liegen auf dem Gateway-Server, nicht auf dem Browser-Rechner.",
    "logDirectory": "Log-Verzeichnis",
    "logDirectoryHelp": "Für das ausgewählte Profil. Die Dateinamen kennzeichnen Kanal und Filter; Dateien entstehen beim Aufzeichnen von Entscheidungen.",
    "contextDirectory": "Verzeichnis für dauerhaften Stufe-1-Kontext",
    "contextDirectoryHelp": "Der neustartfeste private Zustand wird als passive-context.<Kanal>.json nur für den Eigentümer zugänglich gespeichert. Er enthält zurückgehaltene Nachrichtentexte und wird nach erfolgreicher Übergabe an den Agenten gelöscht.",
    "attribution": "Die Groupchat-Koordination wurde von RechnerLotsen entwickelt.",
    "patterns": "Nachrichtenmuster",
    "patternHelp": "Ein regulärer Python-Ausdruck pro Zeile. Änderungen ersetzen die Standards. ^...$ prüft die ganze Nachricht. Leere Listen deaktivieren diese Muster, nicht KI-Prüfung oder Lifecycle-Behandlung. Nur vertrauenswürdige Muster verwenden; komplexe Ausdrücke können die Verarbeitung verlangsamen.",
    "inboundPatterns": "Systemmeldungs-Muster",
    "inboundPatternHelp": "Prüfung am Anfang der ersten Zeile, vor Mention-Routing, mit Groß-/Kleinschreibung. (?i) schaltet diese Unterscheidung aus.",
    "outboundPatterns": "Muster für unterdrückte Antworten",
    "outboundPatternHelp": "Suche ohne Groß-/Kleinschreibung im Original und ohne Markdown, vor Längenschwelle und KI-Prüfung.",
    "multilinePatterns": "Zusätzliche mehrzeilige Prozessmuster",
    "multilineHelp": "Die erste und mindestens eine weitere Zeile müssen ein System- oder Mehrzeilenmuster erfüllen. Diese Muster allein unterdrücken keine einzeiligen Nachrichten.",
    "interruptNotice": "Wörtlicher Text — gesperrte Phrasen",
    "interruptHelp": "Eine wörtliche Phrase pro Zeile (keine Regex). Nur ohne explizite Erwähnung verwerfen, wenn Treffer einer einzelnen Phrase mehr als 50 % der Nachricht ausmachen. Genau 50 % passieren. Groß-/Kleinschreibung wird beachtet; Leerraum wird vereinheitlicht. Verschiedene Phrasen werden nicht addiert. Leere Liste deaktiviert die Prüfung. Zeilen mit # sind Kommentare.",
    "comments": "Zeilen mit # sind Kommentare, z. B. # English keywords. Sie bleiben gespeichert, werden aber nie als Muster ausgewertet. In Regex-Regeln lässt sich ein führendes # mit \\# abgleichen.",
    "restorePatterns": "Standards wiederherstellen",
    "direct": "Mistral direkt",
    "saved": "Gespeichert. Die Änderung wird nach dem Neustart des Gateways dieses Profils wirksam.",
    "primary": "Primär",
    "provider": "Provider",
    "model": "Modell",
    "profileModel": "Leer = Codex-Modell des Profils",
    "modelId": "Modell-ID",
    "eyebrow": "HERMES · ZUSAMMENARBEIT",
    "subtitle": "Relevante Beiträge erkennen. Antworten koordinieren. Bot-Pingpong vermeiden.",
    "profile": "Groupchat-Profil",
    "discardConfirm": "Ungespeicherte Groupchat-Änderungen verwerfen?",
    "dashboardProfile": "Dashboard-Profil",
    "loadError": "Einstellungen konnten nicht geladen werden.",
    "loading": "Einstellungen werden geladen …",
    "enable": "Groupchat aktivieren",
    "enableHint": "Ein gemeinsames Addon für eingehende Relevanz und ausgehenden Pingpong-Schutz.",
    "permissions": "Bestehende Raum-, Benutzer- und Bot-Berechtigungen bleiben gültig. Groupchat schaltet keine zusätzlichen Zugriffe frei.",
    "mentionRequirement": "Für relevanzbasierte Teilnahme an Gruppennachrichten ohne Erwähnung muss Matrix require_mention=false verwenden. Bei true erhält Hermes nur direkte Erwähnungen; der ausgehende Pingpong-Guard funktioniert weiterhin.",
    "participation": "Matrix-Teilnahme",
    "participationHelp": "Gespeicherte effektive Einstellungen aller laufenden benannten Gateways. Aktiv bedeutet: Groupchat und Relevanzfilter sind für Matrix eingeschaltet und require_mention ist false. Änderungen benötigen einen Gateway-Neustart.",
    "activeParticipant": "aktiv",
    "mentionsOnly": "nur Erwähnungen",
    "groupchatOff": "Groupchat aus",
    "filterAI": "Filter-KI",
    "modelHint": "Diese Modellauswahl gilt für beide Filter. Fallbacks werden der Reihe nach versucht. Zugangsdaten bleiben im jeweiligen Hermes-Profil.",
    "addFallback": "+ Fallback hinzufügen",
    "inbound": "EINGANG",
    "relevance": "Relevanzfilter",
    "scoreMessages": "Nachrichten bewerten",
    "relevanceHint": "Direkte Erwähnungen werden sofort übergeben. Andere Beiträge werden nach Relevanz gepuffert; Prozessmeldungen werden ausgefiltert.",
    "delays": "Wartezeit nach Relevanz (Sekunden)",
    "scoreSemantics": "Stufe 0 wird verworfen. Stufe 1 bleibt ohne Timer passiver Kontext. Nur Stufen 2–5 können einen Agentenlauf auslösen; der Filter bewertet immer die neueste Nachricht und verwendet frühere Nachrichten nur als Kontext.",
    "infoDelay": "Nur-zur-Info-Verzögerung (Sekunden)",
    "maxBuffer": "Maximale Nachrichten im Puffer",
    "outbound": "AUSGANG",
    "guard": "Pingpong-Guard",
    "filterReplies": "Antworten filtern",
    "guardHint": "Erkannte Floskeln und Stille-Marker werden unterdrückt. Kurze, uneindeutige Antworten prüft die gemeinsame Filter-KI mit dem Gesprächskontext.",
    "minChars": "Ab dieser Zeichenanzahl ohne KI-Prüfung senden",
    "failOpen": "Bei einem Ausfall der Ausgangsprüfung wird gesendet. Auf jedem ausgewählten Kanal werden Antworten bei aktiviertem Guard vor dem Veröffentlichen gesammelt.",
    "unsaved": "Ungespeicherte Änderungen",
    "loaded": "Einstellungen geladen · Änderungen erfordern einen Gateway-Neustart",
    "discard": "Verwerfen",
    "saving": "Speichert …",
    "save": "Speichern",
    "fallback": "Fallback {number}",
    "removeFallback": "Fallback {number} entfernen",
    "score": "Stufe {number}"
  }
};
  function translator(locale) {
    const language = String(locale || "en").toLowerCase().split(/[-_]/)[0];
    return (key, variables = {}) => {
      const text = messages[language]?.[key] ?? messages.en[key] ?? key;
      return text.replace(/\{(\w+)\}/g, (match, name) => String(variables[name] ?? match));
    };
  }

  function Groupchat() {
    const { locale } = useI18n();
    const t = translator(locale);
    const providers = providerChoices.map(([id, label]) => [id, id === "mistral" ? t("direct") : label]);
    const [profile, setProfile] = useState("current");
    const [profiles, setProfiles] = useState([]);
    const [draft, setDraft] = useState(null);
    const [saved, setSaved] = useState(null);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState("");
    const [error, setError] = useState("");
    const [defaults, setDefaults] = useState(null);
    const [decisionLogDirectory, setDecisionLogDirectory] = useState("");
    const [platformChoices, setPlatformChoices] = useState([]);
    const [participation, setParticipation] = useState([]);
    const [persistentContextDirectory, setPersistentContextDirectory] = useState("");
    useEffect(() => { api.getProfiles().then(result => setProfiles(result.profiles || [])).catch(() => {}); }, []);
    useEffect(() => {
      let cancelled = false;
      setDraft(null); setSaved(null); setError(""); setMessage(""); setDecisionLogDirectory(""); setPersistentContextDirectory("");
      fetchJSON(endpoint(profile)).then(result => {
        if (!cancelled) { setDraft(result.settings); setSaved(result.settings); setDefaults(result.defaults); setDecisionLogDirectory(result.decision_log_directory || ""); setPlatformChoices(result.available_platforms || result.settings.platforms); setParticipation(result.groupchat_participation || []); setPersistentContextDirectory(result.persistent_context_directory || ""); }
      }).catch(err => { if (!cancelled) setError(String(err.message || err)); });
      return () => { cancelled = true; };
    }, [profile]);
    function update(section, key, value) {
      setDraft(previous => section
        ? { ...previous, [section]: { ...previous[section], [key]: value } }
        : { ...previous, [key]: value });
      setMessage("");
    }
    async function save() {
      setBusy(true); setError(""); setMessage("");
      try {
        const settings = { ...draft, relevance: { ...draft.relevance }, pingpong_guard: { ...draft.pingpong_guard } };
        for (const [section, key] of [["relevance", "system_patterns"], ["relevance", "multiline_patterns"], ["relevance", "literal_phrases"], ["pingpong_guard", "silence_patterns"]]) {
          if (Array.isArray(settings[section][key])) settings[section][key] = settings[section][key].filter(line => line.trim());
        }
        await fetchJSON(endpoint(profile), { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ settings }) });
        const refreshed = await fetchJSON(endpoint(profile));
        setParticipation(refreshed.groupchat_participation || []);
        setDraft(settings); setSaved(settings);
        setMessage("saved");
      } catch (err) { setError(String(err.message || err)); }
      finally { setBusy(false); }
    }
    const changed = draft && JSON.stringify(draft) !== JSON.stringify(saved);
    const toggle = (label, checked, change, hint) => h("label", { className: "gc-toggle" },
      h("span", null, h("strong", null, label), hint && h("small", null, hint)),
      h("input", { type: "checkbox", checked, onChange: e => change(e.target.checked), disabled: busy }));
    const number = (section, key, label, min, max) => h("label", { className: "gc-field", key },
      h("span", null, label), h("input", { type: "number", min, max, value: draft[section][key],
        onChange: e => update(section, key, Number(e.target.value)), disabled: busy }));
    const modelRow = (choice, index) => {
      const primary = index === -1;
      function patch(key, value) {
        if (primary) update("filter_model", key, value);
        else update("filter_model", "fallbacks", draft.filter_model.fallbacks.map((item, i) => i === index ? { ...item, [key]: value } : item));
      }
      return h("div", { className: "gc-model", key: index },
        h("div", { className: "gc-model-order" }, primary ? t("primary") : t("fallback", { number: index + 1 })),
        h("label", { className: "gc-field" }, h("span", null, t("provider")),
          h("select", { value: choice.provider, disabled: busy, "aria-label": (primary ? t("primary") : t("fallback", { number: index + 1 })) + " " + t("provider"), onChange: e => patch("provider", e.target.value) }, providers.map(([value, label]) => h("option", { key: value, value }, label)))),
        h("label", { className: "gc-field" }, h("span", null, t("model")),
          h("input", { value: choice.model, disabled: busy, "aria-label": (primary ? t("primary") : t("fallback", { number: index + 1 })) + " " + t("model"), placeholder: choice.provider === "openai-codex" ? t("profileModel") : t("modelId"), onChange: e => patch("model", e.target.value) })),
        !primary && h("button", { className: "gc-icon-button", type: "button", disabled: busy, "aria-label": t("removeFallback", { number: index + 1 }), onClick: () => update("filter_model", "fallbacks", draft.filter_model.fallbacks.filter((_, i) => i !== index)) }, "×"));
    };
    const patternEditor = (section, key, label, hint, literal = false) => h("div", { className: "gc-pattern-editor" },
      h("label", { className: "gc-field" }, h("span", null, t(label)),
        h("textarea", { rows: literal ? 3 : 8, spellCheck: false, disabled: busy,
          value: literal ? (draft[section][key] || "") : (draft[section][key] || []).join("\n"),
          onChange: e => update(section, key, literal ? e.target.value : e.target.value.split("\n")) })),
      h("p", { className: "gc-note" }, t(hint)),
      h("button", { type: "button", className: "gc-secondary", disabled: busy || !defaults,
        onClick: () => update(section, key, defaults[section][key]) }, t("restorePatterns")));
    return h("div", { className: "gc-page" },
      h("header", { className: "gc-header" }, h("div", null,
        h("div", { className: "gc-eyebrow" }, t("eyebrow")),
        h("h1", null, "Groupchat"),
        h("p", null, t("subtitle"))),
        h("label", { className: "gc-field gc-profile" }, h("span", null, t("profile")),
          h("select", { value: profile, disabled: busy, onChange: e => { if (!changed || window.confirm(t("discardConfirm"))) setProfile(e.target.value); } },
            h("option", { value: "current" }, t("dashboardProfile")),
            profiles.filter(p => p.name !== "current").map(p => h("option", { key: p.name, value: p.name }, p.name))))),
      error && h("div", { role: "alert", className: "gc-error" }, error),
      !draft ? h("p", { role: "status" }, error ? t("loadError") : t("loading")) : h(React.Fragment, null,
        h("section", { className: "gc-card" },
          toggle(t("enable"), draft.enabled, value => update(null, "enabled", value), t("enableHint")),
          h("div", { className: "gc-platforms", style: { flexWrap: "wrap" } }, (platformChoices || []).map(platform => h("label", { key: platform },
            h("input", { type: "checkbox", checked: draft.platforms.includes(platform), disabled: busy, onChange: e => update(null, "platforms", e.target.checked ? [...draft.platforms, platform] : draft.platforms.filter(p => p !== platform)) }),
            platform[0].toUpperCase() + platform.slice(1)))),
          h("p", { className: "gc-note" }, t("permissions"))),
        h("section", { className: "gc-card" }, h("h2", null, t("participation")),
          h("p", null, t("mentionRequirement")),
          h("p", { className: "gc-note" }, t("participationHelp")),
          h("div", { className: "gc-platforms", style: { flexWrap: "wrap" } }, participation.map(item =>
            h("span", { key: item.profile }, h("code", null, item.profile), " · require_mention=", String(item.require_mention), " · ",
              item.participates ? t("activeParticipant") : (item.groupchat_enabled ? t("mentionsOnly") : t("groupchatOff")))))),
        h("section", { className: "gc-card" }, h("h2", null, t("filterAI")),
          h("p", null, t("modelHint")),
          modelRow(draft.filter_model, -1), draft.filter_model.fallbacks.map(modelRow),
          h("button", { type: "button", className: "gc-secondary", disabled: busy || draft.filter_model.fallbacks.length >= 5, onClick: () => update("filter_model", "fallbacks", [...draft.filter_model.fallbacks, { provider: "openrouter", model: "" }]) }, t("addFallback"))),
        h("div", { className: "gc-columns" },
          h("section", { className: "gc-card" }, h("div", { className: "gc-eyebrow" }, t("inbound")), h("h2", null, t("relevance")),
            toggle(t("scoreMessages"), draft.relevance.enabled, value => update("relevance", "enabled", value)),
            h("p", null, t("relevanceHint")),
            h("p", { className: "gc-note" }, t("scoreSemantics")),
            h("fieldset", { className: "gc-delays" }, h("legend", null, t("delays")), [5, 4, 3, 2].map(score => h("label", { key: score }, h("span", null, t("score", { number: score })), h("input", { type: "number", min: 0, max: 86400, disabled: busy, value: draft.relevance.score_delays[score], onChange: e => update("relevance", "score_delays", { ...draft.relevance.score_delays, [score]: Number(e.target.value) }) })))),
            number("relevance", "info_delay", t("infoDelay"), 0, 86400),
            number("relevance", "max_context_messages", t("maxBuffer"), 1, 100)),
          h("section", { className: "gc-card" }, h("div", { className: "gc-eyebrow" }, t("outbound")), h("h2", null, t("guard")),
            toggle(t("filterReplies"), draft.pingpong_guard.enabled, value => update("pingpong_guard", "enabled", value)),
            h("p", null, t("guardHint")),
            number("pingpong_guard", "min_chars", t("minChars"), 1, 4000),
            h("p", { className: "gc-note" }, t("failOpen")))),
        h("section", { className: "gc-card gc-patterns" }, h("h2", null, t("patterns")),
          h("p", null, t("patternHelp")),
          h("p", null, t("comments")),
          h("div", { className: "gc-columns" },
            h("div", null, h("h3", null, t("relevance")),
              patternEditor("relevance", "system_patterns", "inboundPatterns", "inboundPatternHelp"),
              patternEditor("relevance", "multiline_patterns", "multilinePatterns", "multilineHelp"),
              patternEditor("relevance", "literal_phrases", "interruptNotice", "interruptHelp")),
            h("div", null, h("h3", null, t("guard")),
              h("p", null, t("languages")),
              patternEditor("pingpong_guard", "silence_patterns", "outboundPatterns", "outboundPatternHelp")))),
        h("section", { className: "gc-card" }, h("h2", null, t("logs")),
          h("p", null, t("logsHelp")),
          h("p", null, t("logDirectory"), h("br"), h("code", { style: { overflowWrap: "anywhere" } }, decisionLogDirectory)),
          h("p", null, t("logDirectoryHelp")),
          h("p", null, t("contextDirectory"), h("br"), h("code", { style: { overflowWrap: "anywhere" } }, persistentContextDirectory)),
          h("p", null, t("contextDirectoryHelp")),
          h("p", { className: "gc-note" }, t("logsRetention"))),
        h("footer", { className: "gc-footer" },
          h("div", null,
            h("span", { role: "status" }, (message && t(message)) || (changed ? t("unsaved") : t("loaded"))),
            h("br"),
            h("a", { href: "https://rechnerlotsen.com/", target: "_blank", rel: "noreferrer" }, t("attribution"))),
          h("button", { type: "button", className: "gc-secondary", disabled: busy || !changed, onClick: () => { setDraft(saved); setMessage(""); } }, t("discard")),
          h("button", { type: "button", className: "gc-primary", disabled: busy || !changed, onClick: save }, busy ? t("saving") : t("save")))));
  }
  window.__HERMES_PLUGINS__.register("groupchat", Groupchat);
})();
