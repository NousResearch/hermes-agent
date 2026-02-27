# Coin Shorts Bot (Community Showcase)

A small automation that generates 1080x1920 YouTube Shorts videos for coins listed in `coins.json`.

## Features
- English text from CoinGecko description
- English voiceover via Edge TTS
- Subtitles at the bottom
- Coin logo overlay
- Output: mp4 files

## Repo
https://github.com/CMZS4/coin-shorts-bot

## Run (quick)
sudo apt update
sudo apt install -y ffmpeg python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python make_coin_short.py --count 5
