import re
import yaml
from pathlib import Path

REPO_ROOT = Path(r"c:\Users\mihai\AppData\Local\hermes\hermes-agent")

DESKTOP_YAML = REPO_ROOT / "locales" / "ro.desktop.yaml"
RO_YAML = REPO_ROOT / "locales" / "ro.yaml"
TUI_YAML = REPO_ROOT / "locales" / "ro.tui.yaml"

desktop_replacements = [
    # Alergarea -> Rulează / Execuție
    (re.compile(r'^(agents\.running:\s*)"Alergarea"', re.M), r'\1"Rulează"'),
    (re.compile(r'^(assistant\.tool\.actions\.running:\s*)"Alergarea"', re.M), r'\1"Rulează"'),
    (re.compile(r'^(assistant\.tool\.statusRunning:\s*)"Alergarea"', re.M), r'\1"Rulează"'),
    (re.compile(r'^(assistant\.tool\.titles\.terminal\.pendingAction:\s*)"Alergarea"', re.M), r'\1"Rulează"'),
    (re.compile(r'^(cron\.control\.failedStop:\s*)"Nu s-a putut opri alergarea"', re.M), r'\1"Nu s-a putut opri execuția"'),
    (re.compile(r'^(cron\.control\.finishedFirst:\s*)"Alergarea s-a terminat înainte să se poată opri"', re.M), r'\1"Execuția s-a terminat înainte să se poată opri"'),
    (re.compile(r'^(settings\.gateway\.webAccess\.running:\s*)"Alergarea"', re.M), r'\1"Rulează"'),
    (re.compile(r'^(settings\.localModels\.serverRunning:\s*)"Alergarea"', re.M), r'\1"Rulează"'),
    (re.compile(r'^(shell\.statusbar\.turnRunning:\s*)"Alergarea"', re.M), r'\1"Rulează"'),
    (re.compile(r'^(statusStack\.running:\s*)"Alergarea"', re.M), r'\1"Rulează"'),

    # Bușteni -> Jurnale
    (re.compile(r'^(sidebar\.logs:\s*)"Bușteni"', re.M), r'\1"Jurnale"'),

    # Tokens / Jetoane
    (re.compile(r'^(commandCenter\.dailyTokens:\s*)"Jetoane zilnice"', re.M), r'\1"Tokeni zilnici"'),
    (re.compile(r'^(commandCenter\.statTokens:\s*)"Jetoane de intrare/ieșire"', re.M), r'\1"Tokeni intrare/ieșire"'),
    (re.compile(r'^(connectorsPage\.dialog\.tokensPerCall:\s*)"jetoane per apel"', re.M), r'\1"tokeni per apel"'),
    (re.compile(r'^(freeTier\.finishingBody:\s*)"Aprobat în browser\. Colectarea jetoanelor de cont\."', re.M), r'\1"Aprobat în browser. Preluarea tokenilor de cont."'),
    (re.compile(r'^(modelPicker\.priceTitle:\s*)"Preț de intrare/ieșire per milion de jetoane"', re.M), r'\1"Preț intrare/ieșire per milion de tokeni"'),
    (re.compile(r'per milion de jetoane', re.M), r'per milion de tokeni'),
    (re.compile(r'jetoanele gateway', re.M), r'tokenii gateway'),
    (re.compile(r'^(shell\.statusbar\.contextUsagePanel\.tokenSummary:\s*)"Jetoane ({[0-9]+} / {[0-9]+})"', re.M), r'\1"Tokeni \2"'),
    (re.compile(r'^(shell\.statusbar\.toggleTokensPerSecond:\s*)"Jetoane pe secundă"', re.M), r'\1"Tokeni pe secundă"'),
    (re.compile(r'^(shell\.statusbar\.tokensPerSecondTitle:\s*)"Jetoane de ieșire pe secundă, în medie pentru ultimele 10 apeluri"', re.M), r'\1"Tokeni de ieșire pe secundă, în medie pentru ultimele 10 apeluri"'),
    (re.compile(r'^(sidebar\.filter\.tokens:\s*)"Jetoane"', re.M), r'\1"Tokeni"'),
]

ro_replacements = [
    (re.compile(r'^(terminal:\s*)"Alergarea"', re.M), r'\1"Rulează"'),
    (re.compile(r'^(context_label:\s*)"Context: jetoane {tokens}"', re.M), r'\1"Context: tokeni {tokens}"'),
    (re.compile(r'^(max_output_label:\s*)"Ieșire maximă: jetoane {tokens}"', re.M), r'\1"Ieșire maximă: tokeni {tokens}"'),
    (re.compile(r'^(window:\s*)"Fereastra: jetoane {total}"', re.M), r'\1"Fereastră: tokeni {total}"'),
    (re.compile(r'^(headroom:\s*)"Margine de limitare: jetoane {headroom}"', re.M), r'\1"Margine de siguranță: tokeni {headroom}"'),
    (re.compile(r'^(category_row:\s*)"{glyph} {label} ~{tokens} jetoane ~{pct}%"', re.M), r'\1"{glyph} {label} ~{tokens} tokeni ~{pct}%"'),
    (re.compile(r'^(toolset_row:\s*)"{toolset} {count} instrumente ~{tokens} jetoane"', re.M), r'\1"{toolset} {count} instrumente ~{tokens} tokeni"'),
    (re.compile(r'^(skill_row:\s*)"{name} index ~{index_tokens} SKILL\.md Jetoane {md_tokens}"', re.M), r'\1"{name} index ~{index_tokens} SKILL.md Tokeni {md_tokens}"'),
    (re.compile(r'^(window_line:\s*)"Fereastra contextuală: jetoane {mark}{used} / {max} \({mark}{pct}%\)"', re.M), r'\1"Fereastră de context: tokeni {mark}{used} / {max} ({mark}{pct}%)"'),
    (re.compile(r'^(label_input_tokens:\s*)"Jetoane de intrare: {count}"', re.M), r'\1"Tokeni de intrare: {count}"'),
    (re.compile(r'^(label_cache_read:\s*)"Jetoane de citire în cache: {count}"', re.M), r'\1"Tokeni citiți din cache: {count}"'),
    (re.compile(r'^(label_cache_write:\s*)"Jetoane de scriere în cache: {count}"', re.M), r'\1"Tokeni scriși în cache: {count}"'),
    (re.compile(r'^(label_output_tokens:\s*)"Jetoane de ieșire: {count}"', re.M), r'\1"Tokeni de ieșire: {count}"'),
    (re.compile(r'^(label_estimated_context:\s*)"Contextul estimat: jetoane ~{count}"', re.M), r'\1"Context estimat: tokeni ~{count}"'),
    (re.compile(r'^(info_context:\s*)"◆ Context: jetoane {tokens} \({source}\)"', re.M), r'\1"◆ Context: tokeni {tokens} ({source})"'),
    (re.compile(r'^(exchange_failed:\s*)"❌ Schimbul de jetoane a eșuat', re.M), r'\1"❌ Schimbul de tokeni a eșuat'),
    (re.compile(r'^(context_refs:\s*)"\[@ context: ref\(e\) {count}, jetoane {tokens}\]"', re.M), r'\1"[@ context: ref(e) {count}, tokeni {tokens}]"'),
    (re.compile(r'^(ctx_tokens_used:\s*)"Jetoane {left}{used} / {max} utilizate"', re.M), r'\1"Tokeni {left}{used} / {max} utilizați"'),
    (re.compile(r'^(context_too_low:\s*)"⚠️ Lungimea contextului este doar jetoane {tokens} — probabil că este prea mică', re.M), r'\1"⚠️ Lungimea contextului este de doar {tokens} tokeni — probabil că este prea mică'),
    (re.compile(r'^(context_minimum:\s*)"Hermes are nevoie de cel puțin jetoane {tokens}\.', re.M), r'\1"Hermes are nevoie de cel puțin {tokens} tokeni.'),
    (re.compile(r'^(token_provider:\s*)"Furnizor de jetoane"', re.M), r'\1"Furnizor de tokeni"'),
    (re.compile(r'retrimiteți jetoane de intrare complete', re.M), r'retrimiteți tokeni de intrare compleți'),
    (re.compile(r'^(label_input_tokens:\s*)"Jetoane de intrare:"', re.M), r'\1"Tokeni de intrare:"'),
    (re.compile(r'^(label_output_tokens:\s*)"Jetoane de ieșire:"', re.M), r'\1"Tokeni de ieșire:"'),
    (re.compile(r'^(label_prompt_tokens_total:\s*)"Jetoane prompte \(total\):"', re.M), r'\1"Tokeni de prompt (total):"'),
    (re.compile(r'^(label_completion_tokens:\s*)"Jetoane de finalizare:"', re.M), r'\1"Tokeni de completare:"'),
    (re.compile(r'^(label_total_tokens:\s*)"Total de jetoane:"', re.M), r'\1"Total tokeni:"'),
    (re.compile(r'\(\(jetoane ~{tokens}\)\)', re.M), r'((~{tokens} tokeni))'),
    (re.compile(r'\(jetoane ~{tokens}\)', re.M), r'(~{tokens} tokeni)'),
    (re.compile(r'pentru a vedea jetoanele care apar în timp real', re.M), r'pentru a vedea tokenii care apar în timp real'),
]

tui_replacements = [
    (re.compile(r'^(chatBits\.thinking\.approxTokens:\s*)"Jetoane ~{0}"', re.M), r'\1"Tokeni ~{0}"'),
    (re.compile(r'^(hubs\.agents\.detail\.subtreeTokens:\s*)"jetoane subarbore"', re.M), r'\1"tokeni subarbore"'),
    (re.compile(r'^(hubs\.agents\.detail\.tokens:\s*)"jetoane"', re.M), r'\1"tokeni"'),
    (re.compile(r'^(hubs\.agents\.diff\.tokens:\s*)"jetoane"', re.M), r'\1"tokeni"'),
    (re.compile(r'^(slashCmd\.core\.logs\.pageTitle:\s*)"Bușteni"', re.M), r'\1"Jurnale"'),
    (re.compile(r'^(slashCmd\.session\.usage\.rowInputTokens:\s*)"Jetoane de intrare"', re.M), r'\1"Tokeni de intrare"'),
    (re.compile(r'^(slashCmd\.session\.usage\.rowOutputTokens:\s*)"Jetoane de ieșire"', re.M), r'\1"Tokeni de ieșire"'),
    (re.compile(r'^(slashCmd\.session\.usage\.rowTotalTokens:\s*)"Total de jetoane"', re.M), r'\1"Total tokeni"'),
]

def apply_replacements(path, replacements):
    text = path.read_text(encoding="utf-8")
    original = text
    count = 0
    for pattern, repl in replacements:
        new_text, n = pattern.subn(repl, text)
        if n > 0:
            count += n
            text = new_text
    print(f"Applied {count} replacements to {path.name}")
    if text != original:
        path.write_text(text, encoding="utf-8")
        # Validate yaml syntax
        try:
            yaml.safe_load(text)
            print(f"Validated YAML syntax for {path.name} OK.")
        except Exception as e:
            print(f"ERROR: Invalid YAML in {path.name}: {e}")
            path.write_text(original, encoding="utf-8")
            raise

if __name__ == "__main__":
    apply_replacements(DESKTOP_YAML, desktop_replacements)
    apply_replacements(RO_YAML, ro_replacements)
    apply_replacements(TUI_YAML, tui_replacements)
