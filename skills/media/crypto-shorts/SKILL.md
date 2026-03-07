# Crypto Shorts (MP4) Skill

Goal: Generate a 1080×1920 MP4 Shorts video for a given coin with:
- CoinGecko EN description + logo
- English voiceover (Edge TTS)
- Bottom subtitles
- ffmpeg render

## User intent
When the user says:
- "make a crypto video about Bitcoin"
- "create a short video for Solana"
- "generate a crypto shorts mp4 for ETH"

…use this skill. The user should NOT need to provide paths/commands.

## Defaults (no user input required)
- Default output directory: `~/out/crypto-shorts/`
- If coin is not specified: ask once. Otherwise use the coin from the request.

## Requirements (WSL Ubuntu)
terminal(command="sudo apt update && sudo apt install -y ffmpeg python3 python3-venv python3-pip", timeout=600)
terminal(command="python3 -m pip install --upgrade pip && python3 -m pip install edge-tts", timeout=600)

## Steps (agent should run these via terminal tool)
1) Ensure output directory exists
terminal(command="mkdir -p ~/out/crypto-shorts", timeout=60)

2) Generate 1 video (replace COIN with the requested coin, e.g. bitcoin / ethereum / solana)
terminal(command="python3 SKILL_DIR/scripts/make_crypto_short.py --coin COIN --out_dir ~/out/crypto-shorts", timeout=900)

3) Verify the newest MP4
terminal(command="ls -lt ~/out/crypto-shorts/*.mp4 | head -n 3", timeout=60)

## Success criteria
- A new `.mp4` exists in `~/out/crypto-shorts/`
- Agent returns the newest mp4 filename/path to the user

## Notes
- CoinGecko is public/no API key; may rate-limit (retry after 30–60s).
- If `edge-tts` fails, ensure internet access and that it is installed.
