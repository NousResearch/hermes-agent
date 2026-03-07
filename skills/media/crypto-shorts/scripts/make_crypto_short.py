#!/usr/bin/env python3
import argparse, datetime, json, re, subprocess, urllib.parse, urllib.request
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
UA = "hermes-agent-skill-crypto-shorts/1.0"
BASE = "https://api.coingecko.com/api/v3"

def http_json(path: str, params: dict | None = None) -> dict:
    qs = ""
    if params:
        qs = "?" + urllib.parse.urlencode(params)
    url = f"{BASE}{path}{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", errors="ignore"))

def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", s).strip()

def first_sentences(text: str, n=2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:n]).strip()

def resolve_id(query: str) -> str:
    data = http_json("/search", {"query": query})
    coins = data.get("coins", [])
    return coins[0]["id"] if coins else ""

def fetch_coin(coin_id: str) -> dict:
    data = http_json(
        f"/coins/{urllib.parse.quote(coin_id)}",
        {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false",
        },
    )
    desc = strip_html((data.get("description") or {}).get("en", ""))
    img = ((data.get("image") or {}).get("large")) or ""
    name = data.get("name") or coin_id
    symbol = (data.get("symbol") or "").upper()
    # sanitize for ffmpeg drawtext safety
    symbol_safe = re.sub(r"[^A-Z0-9]", "", symbol)
    return {"id": coin_id, "name": name, "symbol": symbol_safe, "desc": desc, "image": img}

def download(url: str, path: Path) -> bool:
    if not url:
        return False
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        path.write_bytes(r.read())
    return path.exists() and path.stat().st_size > 500

def build_script(name: str, symbol: str, desc: str) -> str:
    base = first_sentences(desc, 2)
    if base:
        w = base.split()
        if len(w) > 42:
            base = " ".join(w[:42]).strip() + "."
    s1 = f"{name} ({symbol}) is a crypto project in the ecosystem."
    s2 = base if base else "It is designed for specific use cases within blockchain networks."
    s3 = "Risk note: crypto assets are volatile and can involve technical, security, and regulatory risks."
    s4 = "This is not financial advice."
    text = " ".join([s1, s2, s3, s4])
    w = text.split()
    if len(w) > 95:
        text = " ".join(w[:95]).strip()
        if "not financial advice" not in text.lower():
            text += " This is not financial advice."
    return text

def srt_time(ms:int)->str:
    h=ms//3600000; ms%=3600000
    m=ms//60000; ms%=60000
    s=ms//1000; ms%=1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def make_srt(text:str, total_ms:int=34000)->str:
    words = text.split()
    step = 5
    chunks = [" ".join(words[i:i+step]) for i in range(0, len(words), step)]
    seg = max(1200, total_ms // max(1, len(chunks)))
    cur=0; out=[]
    for i,ch in enumerate(chunks,1):
        start=cur; end=min(total_ms, cur+seg)
        out += [str(i), f"{srt_time(start)} --> {srt_time(end)}", ch, ""]
        cur=end
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True, help="Coin name/symbol/id (e.g. bitcoin, BTC, ethereum)")
    ap.add_argument("--out_dir", default="out", help="Output directory for mp4")
    ap.add_argument("--voice", default="en-US-JennyNeural", help="Edge TTS voice")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    assets = out_dir / "assets"; assets.mkdir(exist_ok=True)

    coin_id = resolve_id(args.coin) or args.coin.lower()
    info = fetch_coin(coin_id)

    logo_path = assets / f"{coin_id}.png"
    has_logo = logo_path.exists() and logo_path.stat().st_size > 500
    if not has_logo:
        has_logo = download(info["image"], logo_path)

    symbol = info["symbol"] or args.coin.upper()
    text = build_script(info["name"], symbol, info["desc"])

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{symbol}_{stamp}"

    txt = out_dir / f"{base}.txt"
    srt = out_dir / f"{base}.srt"
    mp3 = out_dir / f"{base}.mp3"
    mp4 = out_dir / f"{base}.mp4"
    title_file = out_dir / f"{base}_title.txt"

    title = f"{info['name']} ({symbol}) — What is it?"
    txt.write_text(text, encoding="utf-8")
    srt.write_text(make_srt(text), encoding="utf-8")
    title_file.write_text(title, encoding="utf-8")

    subprocess.check_call(["edge-tts","--voice",args.voice,"--rate","+10%","--file",str(txt),"--write-media",str(mp3)])

    sub_style = "FontName=DejaVu Sans,FontSize=12,Outline=2,Shadow=1,Alignment=2,MarginV=18"

    if has_logo:
        cmd = [
            "ffmpeg","-y",
            "-f","lavfi","-i","color=c=#0b1020:s=1080x1920:r=30:d=60",
            "-loop","1","-i",str(logo_path),
            "-i",str(mp3),
            "-filter_complex",
            (
                f"[0:v]drawtext=fontfile={FONT}:text='{symbol}':fontcolor=white@0.18:fontsize=220:x=(W-text_w)/2:y=700[bg0];"
                "[1:v]scale=720:-1:flags=lanczos,format=rgba,colorchannelmixer=aa=0.92[logo];"
                "[bg0][logo]overlay=x=(W-w)/2:y=560[bg];"
                "[bg]"
                "drawbox=x=0:y=0:w=iw:h=260:color=black@0.35:t=fill,"
                f"drawtext=fontfile={FONT}:textfile='{title_file}':reload=1:fontcolor=white:fontsize=58:borderw=4:bordercolor=black:x=(w-text_w)/2:y=95,"
                f"subtitles='{srt}':force_style='{sub_style}'"
                "[v]"
            ),
            "-map","[v]","-map","2:a",
            "-c:v","libx264","-pix_fmt","yuv420p",
            "-c:a","aac",
            "-shortest",
            str(mp4)
        ]
    else:
        cmd = [
            "ffmpeg","-y",
            "-f","lavfi","-i","color=c=#0b1020:s=1080x1920:r=30:d=60",
            "-i",str(mp3),
            "-vf",
            (
                "drawbox=x=0:y=0:w=iw:h=260:color=black@0.35:t=fill,"
                f"drawtext=fontfile={FONT}:textfile='{title_file}':reload=1:fontcolor=white:fontsize=58:borderw=4:bordercolor=black:x=(w-text_w)/2:y=95,"
                f"subtitles='{srt}':force_style='{sub_style}'"
            ),
            "-c:v","libx264","-pix_fmt","yuv420p",
            "-c:a","aac",
            "-shortest",
            str(mp4)
        ]

    subprocess.check_call(cmd)
    print(str(mp4))

if __name__ == "__main__":
    main()
