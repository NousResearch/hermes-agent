from pathlib import Path
import yaml
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[1]
FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
]

def load_font(size):
    for path in FONT_CANDIDATES:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()

def wrap(draw, text, font, width):
    words = list(text)
    lines = []
    cur = ''
    for ch in words:
        test = cur + ch
        if draw.textlength(test, font=font) <= width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines

def render_episode(ep_dir: Path):
    scroll = yaml.safe_load((ep_dir/'scroll_plan.yaml').read_text(encoding='utf-8'))
    lettering = yaml.safe_load((ep_dir/'lettering_script.yaml').read_text(encoding='utf-8'))
    out_dir = BASE / 'renders' / ep_dir.name
    panel_dir = out_dir / 'panels'
    panel_dir.mkdir(parents=True, exist_ok=True)
    width = 1080
    panel_h = 760
    title_font = load_font(44)
    body_font = load_font(30)
    small_font = load_font(24)
    panel_paths = []
    captions = {c['panel_id']: c['text'] for c in lettering.get('captions', [])}
    balloons = {}
    for item in lettering.get('balloons', []):
        balloons.setdefault(item['panel_id'], []).append(item['text'])
    colors = [(33,40,70),(73,33,70),(30,68,64),(68,43,30),(45,45,45),(80,51,91),(26,55,93),(90,33,52)]
    for idx, block in enumerate(scroll['blocks'], start=1):
        panel = Image.new('RGB', (width, panel_h), colors[(idx-1)%len(colors)])
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((40,40,width-40,panel_h-40), radius=36, outline=(255,255,255), width=3)
        draw.text((70,70), f"{ep_dir.name.upper()} / {block['block_id']} / {block['emotion']}", font=small_font, fill=(230,230,230))
        visual = block['visual']
        visual_lines = wrap(draw, visual, body_font, width-140)
        y = 150
        for line in visual_lines:
            draw.text((70,y), line, font=body_font, fill=(255,255,255))
            y += 42
        pid = f'p{idx:02d}'
        if pid in captions:
            draw.rounded_rectangle((70, y+30, width-70, y+130), radius=26, fill=(18,18,18))
            for i, line in enumerate(wrap(draw, captions[pid], body_font, width-140)):
                draw.text((100, y+55 + i*38), line, font=body_font, fill=(255,255,255))
            y += 150
        for bidx, text in enumerate(balloons.get(pid, []), start=1):
            top = y + 20 + (bidx-1)*120
            draw.rounded_rectangle((120, top, width-120, top+90), radius=45, fill=(245,245,245))
            for i, line in enumerate(wrap(draw, text, body_font, width-300)):
                draw.text((170, top+22+i*34), line, font=body_font, fill=(25,25,25))
        draw.text((70,panel_h-90), f"purpose: {block['purpose']}", font=small_font, fill=(220,220,220))
        panel_path = panel_dir / f'{pid}.png'
        panel.save(panel_path)
        panel_paths.append(panel_path)
    gap = 40
    total_h = len(panel_paths)*panel_h + max(0, len(panel_paths)-1)*gap + 120
    longscroll = Image.new('RGB', (width, total_h), (245,245,245))
    y = 60
    for p in panel_paths:
        img = Image.open(p).convert('RGB')
        longscroll.paste(img, (0,y))
        y += panel_h + gap
    longscroll_path = out_dir / f'{ep_dir.name}_longscroll.png'
    longscroll.save(longscroll_path)
    return {'episode': ep_dir.name, 'panel_count': len(panel_paths), 'longscroll': str(longscroll_path)}

if __name__ == '__main__':
    outputs = []
    for ep_dir in sorted((BASE/'webtoon').glob('ep*')):
        outputs.append(render_episode(ep_dir))
    import json
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
