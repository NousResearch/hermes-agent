"""Runtime presentation of built-in command descriptions.

Registry definitions and category identifiers remain stable. Resolve language at
presentation time so imports and cached command maps cannot freeze a profile's
language. Commands without a translation retain their original description.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.commands import CommandDef


SWEDISH_DESCRIPTIONS = {
    "start": "Bekräfta plattformens startsignaler utan att skicka ett svar",
    "new": "Starta en ny session (nytt sessions-ID och ny historik)",
    "topic": "Aktivera eller granska ämnessessioner i Telegram-direktmeddelanden",
    "clear": "Rensa skärmen och starta en ny session",
    "redraw": "Rita om hela gränssnittet (återställ terminalens visning)",
    "history": "Visa samtalshistoriken",
    "save": "Exportera det aktuella samtalet (/save utan argument visar användningen)",
    "retry": "Försök igen med det senaste meddelandet (skicka om till agenten)",
    "prompt": "Skriv nästa prompt i $EDITOR (Markdown) och skicka den sedan",
    "undo": "Gå tillbaka N användarturer och skriv en ny prompt (standard: 1)",
    "title": "Ange en titel för den aktuella sessionen",
    "handoff": "Lämna över sessionen till en meddelandeplattform (Telegram, Discord med flera)",
    "branch": "Förgrena den aktuella sessionen (utforska en annan väg)",
    "worktree": "Visa, lista, skapa eller rensa isolerade Git-arbetskataloger",
    "compress": "Komprimera samtalskontexten ('here [N]' behåller de senaste N turerna; --preview visar effekten)",
    "rollback": "Lista eller återställ filsystemets kontrollpunkter (behåller dina ändringar; --all åsidosätter)",
    "snapshot": "Skapa eller återställ ögonblicksbilder av Hermes konfiguration och tillstånd",
    "export": "Exportera en profil (konfiguration, färdigheter, tema) till ett delbart arkiv",
    "import": "Importera ett delat profilarkiv som en ny profil",
    "stop": "Stoppa alla pågående bakgrundsprocesser",
    "pause": "Pausa nytt arbete globalt (nödstopp); '/pause off' återupptar",
    "approve": "Godkänn ett väntande farligt kommando",
    "deny": "Neka ett väntande farligt kommando (valfritt med en motivering)",
    "bg": "Kör en prompt i en separat bakgrundssession",
    "btw": "Ställ en sidofråga om det aktuella samtalet utan att avbryta det",
    "agents": "Visa aktiva agenter och pågående uppgifter",
    "journey": "Öppna tidslinjen för inlärning",
    "queue": "Köa en prompt till nästa tur (avbryter inte)",
    "steer": "Infoga ett meddelande efter nästa verktygsanrop utan att avbryta",
    "goal": "Ange ett varaktigt mål som Hermes arbetar mot över flera turer tills det är uppnått",
    "heartbeat": "Ange en återkommande prompt som återgår till sessionen när den är inaktiv",
    "refine": "Granska samtalet nu och spara lärdomar i minnet eller som färdigheter",
    "review": "Starta en oberoende underagent för att granska det nyss diskuterade arbetet (PR, kod, dokumentation)",
    "loop": "Kör en prompt igen med ett återkommande intervall i sessionen",
    "plan": "Skriv en genomförandeplan i Markdown till .hermes/plans/ utan att utföra något",
    "moa": "Kör en prompt med standardförvalet för Mixture of Agents och återställ sedan modellen",
    "subgoal": "Lägg till eller hantera extra kriterier för det aktiva målet",
    "status": "Visa information om session, modell, token och kontext",
    "egress": "Visa status för Dockers proxy för utgående trafik",
    "context": "Visa kontextfönstret med användningsmätare, kategorifördelning, komprimeringsstatistik och genomströmning",
    "whoami": "Visa din åtkomst till snedstreckskommandon (administratör/användare)",
    "profile": "Visa den aktiva profilens namn och hemkatalog",
    "sethome": "Ange den här chatten som hemkanal",
    "resume": "Återuppta en tidigare namngiven session",
    "sessions": "Bläddra bland och återuppta tidigare sessioner",
    "config": "Visa aktuell konfiguration",
    "model": "Byt modell (gäller sessionen; --global sparar valet)",
    "codex-runtime": "Aktivera eller inaktivera codex app-server för OpenAI/Codex-modeller",
    "personality": "Ange en fördefinierad personlighet",
    "statusbar": "Visa eller dölj statusraden för kontext och modell",
    "battery": "Visa eller dölj en färgkodad batteriindikator i statusraden",
    "timestamps": "Visa eller dölj tidsstämplar [HH:MM] på meddelanden och i /history",
    "diff": "Visa Git-ändringar i arbetskatalogen",
    "verbose": "Växla visning av verktygsförlopp: off -> new -> all -> verbose",
    "focus": "Växla fokusvy – visa endast din prompt och det slutliga svaret",
    "footer": "Visa eller dölj sidfoten med körningsmetadata i slutliga gatewaysvar",
    "yolo": "Växla YOLO-läge (hoppa över alla godkännanden av farliga kommandon)",
    "approvals": "Visa eller ange bestående godkännandeläge för farliga kommandon",
    "reasoning": "Hantera resonemangsnivå och visning",
    "fast": "Snabbläge – OpenAI Priority Processing / Anthropic Fast Mode (normal/fast/auto/cold)",
    "skin": "Visa eller ändra utseende/tema",
    "indicator": "Välj stil för TUI:ns upptagenindikator",
    "voice": "Aktivera eller inaktivera röstläge",
    "wake": "Aktivera eller inaktivera lyssning efter väckningsfrasen 'Hey Hermes'",
    "busy": "Styr hur meddelanden hanteras medan Hermes arbetar",
    "tools": "Hantera verktyg: /tools [list|disable|enable] [name...]",
    "toolsets": "Lista tillgängliga verktygsuppsättningar",
    "skills": "Sök, installera, granska eller hantera färdigheter",
    "memory": "Granska väntande minnesskrivningar eller växla godkännandekravet",
    "bundles": "Lista färdighetspaket (alias /<name> för flera färdigheter)",
    "pet": "Visa, dölj eller adoptera en petdex-maskot (/pet, /pet list, /pet <slug>)",
    "hatch": "Skapa ett nytt petdex-husdjur från en beskrivning",
    "learn": "Lär in en återanvändbar färdighet från det du beskriver (kataloger, URL:er, chatten, anteckningar)",
    "init": "Skapa eller uppdatera projektinstruktioner i AGENTS.md genom att granska kodförrådet",
    "cron": "Hantera schemalagda uppgifter",
    "suggestions": "Granska föreslagna automatiseringar (godkänn/avvisa)",
    "blueprint": "Konfigurera en automatisering från en mall",
    "curator": "Underhåll färdigheter i bakgrunden (status, run, pin, archive, list-archived)",
    "kanban": "Samarbetstavla för flera profiler (uppgifter, länkar, kommentarer)",
    "reload": "Läs in .env-variabler på nytt i den pågående sessionen",
    "reload-mcp": "Läs in MCP-servrar på nytt från konfigurationen",
    "reload-skills": "Sök i ~/.hermes/skills/ efter nyinstallerade eller borttagna färdigheter",
    "browser": "Anslut webbläsarverktyg till din öppna Chromium-baserade webbläsare via CDP eller byt till Browser Use-läge",
    "plugins": "Lista installerade pluginer och deras status",
    "commands": "Bläddra bland alla kommandon och färdigheter (sidindelat)",
    "help": "Visa tillgängliga kommandon (/help skills listar färdighetskommandon, /help <text> filtrerar)",
    "palette": "Öppna kommandopaletten med ungefärlig sökning (även Ctrl+P)",
    "restart": "Starta om gatewayen ordnat när pågående körningar är färdiga",
    "usage": "Visa tokenanvändning och användningsgränser; `reset` löser in en sparad återställning av Codex-gränser",
    "subscription": "Visa ditt Nous-abonnemang och ändra det i webbläsaren",
    "login": "Logga in med ett Nous-konto (behåller dina anslutningar)",
    "topup": "Visa ditt Nous-saldo och hantera fakturering på portalen",
    "insights": "Visa användningsstatistik och analyser",
    "platforms": "Visa status för gateway- och meddelandeplattformar",
    "platform": "Pausa, återuppta eller lista en felande gatewayplattform",
    "copy": "Kopiera assistentens senaste svar till urklipp",
    "paste": "Bifoga en bild från urklipp",
    "image": "Bifoga en lokal bildfil till nästa prompt",
    "update": "Uppdatera Hermes Agent till den senaste versionen",
    "version": "Visa Hermes Agents version",
    "debug": "Ladda upp en felsökningsrapport (systeminformation och loggar) och få delbara länkar",
    "quit": "Avsluta CLI:n (--delete tar även bort sessionshistoriken)",
}


def command_description(cmd: CommandDef) -> str:
    """Translate display copy only; never mutate the shared command definition."""
    from agent.i18n import get_language

    if get_language() == "sv":
        return SWEDISH_DESCRIPTIONS.get(cmd.name, cmd.description)
    return cmd.description


def cli_command_description(name: str, fallback: str) -> str:
    """Translate a canonical or alias entry without changing its argument syntax."""
    from agent.i18n import get_language
    from hermes_cli.commands import resolve_command

    if get_language() != "sv":
        return fallback
    cmd = resolve_command(name)
    if cmd is None or cmd.name not in SWEDISH_DESCRIPTIONS:
        return fallback
    description = SWEDISH_DESCRIPTIONS[cmd.name]
    if name.lstrip("/").lower() != cmd.name:
        return f"{description} (alias för /{cmd.name})"
    if cmd.args_hint:
        return f"{description} (användning: /{cmd.name} {cmd.args_hint})"
    return description
