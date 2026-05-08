<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Dokumentasie"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="Lisensie: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Gebou deur Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-lightgrey?style=for-the-badge" alt="English"></a>
</p>

**Die selfverbeterende agent gebou deur [Nous Research](https://nousresearch.com).** Hermes skep vaardighede uit ervaring, verbeter dit tydens gebruik, hou kennis oor sessies heen by, en kan vanaf die terminale of boodskapplatforms soos Telegram, Discord, Slack, WhatsApp en Signal gebruik word.

Hermes werk met jou gekose modelverskaffer, insluitend [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai), NVIDIA NIM, Xiaomi MiMo, z.ai/GLM, Kimi/Moonshot, MiniMax, Hugging Face, of jou eie eindpunt. Gebruik `hermes model` om te wissel.

## Vinnige installasie

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Hermes ondersteun Linux, macOS, WSL2 en Android via Termux. Inheemse Windows word nie ondersteun nie, gebruik asseblief WSL2.

Na installasie:

```bash
source ~/.bashrc
hermes
```

## Algemene opdragte

```bash
hermes              # Interaktiewe CLI
hermes model        # Kies jou modelverskaffer en model
hermes tools        # Stel gereedskap op
hermes config set   # Stel individuele konfigurasiewaardes
hermes gateway      # Begin die boodskap-gateway
hermes setup        # Loop die volledige opstelling
hermes update       # Dateer Hermes op
hermes doctor       # Diagnoseer probleme
```

## Afrikaans as vertoontaal

Hermes kan statiese gebruikerboodskappe in Afrikaans wys, insluitend CLI-goedkeuringsprompts en sekere gateway-antwoorde.

Stel dit so:

```yaml
display:
  language: af
```

Of vir 'n enkele sessie:

```bash
HERMES_LANGUAGE=af hermes
```

Hierdie instelling vertaal nie modelgegenereerde antwoorde, logs, foutspore, gereedskapuitvoer of slash-opdragname nie. As jy wil hê die agent self moet in Afrikaans antwoord, vra dit in jou prompt of stelselboodskap.

## Dokumentasie

Die volledige dokumentasie is beskikbaar by [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/).

## Bydra

Sien die [Contributing Guide](CONTRIBUTING.md) vir ontwikkelingsopstelling, toetsing en PR-riglyne.
