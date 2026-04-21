from __future__ import annotations

import json
import textwrap
from pathlib import Path

import requests
import yaml
from PIL import Image, ImageDraw, ImageFont
import fal_client

BASE = Path('docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001')
OUT = BASE / 'generated_fal_v2'
OUT.mkdir(parents=True, exist_ok=True)

PANEL_W = 720
PANEL_H = 1072
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
TITLE_FONT = ImageFont.truetype(FONT_PATH, 28)
TEXT_FONT = ImageFont.truetype(FONT_PATH, 28)
SMALL_FONT = ImageFont.truetype(FONT_PATH, 20)
CAPTION_FONT = ImageFont.truetype(FONT_PATH, 24)

with open(BASE / 'panel_prompts.yaml', 'r', encoding='utf-8') as f:
    panel_data = yaml.safe_load(f)
with open(BASE / 'lettering_script.yaml', 'r', encoding='utf-8') as f:
    lettering = yaml.safe_load(f)
with open(BASE / 'scroll_plan.yaml', 'r', encoding='utf-8') as f:
    scroll = yaml.safe_load(f)

panel_map = {p['panel_id']: p for p in panel_data['panels']}
spacing_map = {b['block_id']: b.get('spacing', 'medium') for b in scroll['blocks']}
balloons_by_panel = {}
for b in lettering.get('balloons', []):
    balloons_by_panel.setdefault(b['panel_id'], []).append(b)
captions_by_panel = {}
for c in lettering.get('captions', []):
    captions_by_panel.setdefault(c['panel_id'], []).append(c)


def fal_generate(prompt: str):
    return fal_client.subscribe(
        'fal-ai/flux-2-pro',
        arguments={
            'prompt': prompt,
            'image_size': {'width': PANEL_W, 'height': PANEL_H},
            'num_images': 1,
            'output_format': 'png'
        },
    )['images'][0]['url']


def fal_edit(prompt: str, image_urls: list[str]):
    return fal_client.subscribe(
        'fal-ai/flux-2-pro/edit',
        arguments={
            'prompt': prompt,
            'image_urls': image_urls,
            'image_size': {'width': PANEL_W, 'height': PANEL_H},
            'num_images': 1,
            'output_format': 'png'
        },
    )['images'][0]['url']


def download(url: str, path: Path):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)


def wrap_text(text: str, width: int = 18) -> str:
    return '\n'.join(textwrap.wrap(text, width=width, break_long_words=False))


def rounded_box(draw: ImageDraw.ImageDraw, xy, radius=18, fill=(255,255,255,235), outline=(30,30,30,255), width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def ellipse_balloon(draw: ImageDraw.ImageDraw, xy, fill=(255,255,255,250), outline=(25,25,25,255), width=3):
    draw.ellipse(xy, fill=fill, outline=outline, width=width)


def add_tail(draw: ImageDraw.ImageDraw, tip, base_left, base_right, fill=(255,255,255,250), outline=(25,25,25,255), width=3):
    draw.polygon([base_left, base_right, tip], fill=fill, outline=outline)
    draw.line([base_left, tip, base_right], fill=outline, width=width)


def mask_region(draw: ImageDraw.ImageDraw, xy, fill=(244,244,244,235)):
    draw.rounded_rectangle(xy, radius=18, fill=fill)


BALLOON_LAYOUTS = {
    'p02': [
        {'xy': (405, 52, 690, 210), 'tail_tip': (530, 255), 'base_left': (505, 196), 'base_right': (545, 196)},
    ],
    'p04': [
        {'xy': (402, 70, 690, 205), 'tail_tip': (565, 265), 'base_left': (540, 192), 'base_right': (580, 192)},
    ],
    'p06': [
        {'xy': (405, 55, 690, 180), 'tail_tip': (560, 255), 'base_left': (540, 166), 'base_right': (576, 166)},
    ],
    'p07': [
        {'xy': (100, 70, 405, 210), 'tail_tip': (300, 300), 'base_left': (268, 197), 'base_right': (320, 197)},
    ],
    'p08': [
        {'xy': (410, 50, 690, 180), 'tail_tip': (540, 265), 'base_left': (520, 166), 'base_right': (560, 166)},
        {'xy': (360, 220, 690, 420), 'tail_tip': (545, 505), 'base_left': (518, 405), 'base_right': (566, 405)},
    ],
}

CAPTION_LAYOUTS = {
    'p01': (24, 28, 300, 130),
    'p03': (24, 28, 300, 126),
    'p07': (24, 760, 340, 900),
    'p08': (24, 820, 380, 980),
}

MASK_LAYOUTS = {
    'p03': [(160, 265, 635, 505)],
    'p04': [(110, 245, 650, 430)],
    'p05': [(118, 220, 650, 425)],
    'p07': [(180, 255, 620, 485)],
    'p08': [(175, 265, 610, 520)],
}


def add_text_layer(img_path: Path, panel_id: str):
    img = Image.open(img_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in MASK_LAYOUTS.get(panel_id, []):
        mask_region(draw, box)

    for cap in captions_by_panel.get(panel_id, []):
        x1, y1, x2, y2 = CAPTION_LAYOUTS.get(panel_id, (24, 24, 320, 120))
        rounded_box(draw, (x1, y1, x2, y2), radius=18, fill=(20, 24, 32, 215), outline=(255, 255, 255, 40), width=2)
        txt = wrap_text(cap['text'], 13)
        draw.multiline_text((x1 + 18, y1 + 16), txt, font=CAPTION_FONT, fill=(255, 255, 255, 255), spacing=6)

    for idx, bal in enumerate(balloons_by_panel.get(panel_id, [])):
        layout = BALLOON_LAYOUTS.get(panel_id, [])
        if idx >= len(layout):
            continue
        spec = layout[idx]
        ellipse_balloon(draw, spec['xy'])
        add_tail(draw, spec['tail_tip'], spec['base_left'], spec['base_right'])
        txt = wrap_text(bal['text'], 11 if panel_id in {'p08'} else 12)
        bx1, by1, bx2, by2 = spec['xy']
        bbox = draw.multiline_textbbox((0, 0), txt, font=TEXT_FONT, spacing=6)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = bx1 + ((bx2 - bx1) - tw) / 2
        ty = by1 + ((by2 - by1) - th) / 2 - 4
        draw.multiline_text((tx, ty), txt, font=TEXT_FONT, fill=(15,15,15,255), spacing=6, align='center')

    combined = Image.alpha_composite(img, overlay).convert('RGB')
    combined.save(img_path)


style_prefix = (
    'clean Korean webtoon cartoon illustration, 2D cel shading, expressive line art, softened shapes, '
    'limited color palette, mobile vertical comic panel, polished digital manhwa style, '
    'consistent same apartment study room, same 19-year-old Korean male student in black hoodie, '
    'same Korean mother in simple beige homewear when present, emotionally readable faces, '
    'absolutely no text, no letters, no Korean characters, no English characters, no subtitles, no watermark'
)

location_anchor_prompt = (
    style_prefix + ', empty small Busan apartment study room at night, cluttered desk with mock exam papers, '
    'laptop glow, textbooks stacked, blue-gray shadows, quiet tense atmosphere, cartoon webtoon background art'
)
protagonist_anchor_prompt = (
    style_prefix + ', tired 19-year-old Korean male student, slim build, black hoodie, exhausted eyes, '
    'clean anime-webtoon facial structure, sitting at desk in same room, laptop glow on face'
)
mother_anchor_prompt = (
    style_prefix + ', Korean mother in late 40s, restrained sharp expression, simple beige homewear, '
    'clean manhwa face design, standing in same study room, subtle hallway light contrast'
)

print('Generating anchors...')
location_url = fal_generate(location_anchor_prompt)
protagonist_url = fal_generate(protagonist_anchor_prompt)
mother_url = fal_generate(mother_anchor_prompt)

manifest = {
    'mode': 'fal_flux2_render',
    'anchors': {
        'location': location_url,
        'protagonist': protagonist_url,
        'mother': mother_url,
    },
    'panels': []
}

prev_url = None
rendered_paths = []
for idx in range(1, 9):
    panel_id = f'p{idx:02d}'
    panel = panel_map[panel_id]
    prompt = f"{style_prefix}, {panel['prompt']}, comic panel composition, clean speech-balloon-safe negative space, no visible screen text"
    refs = [location_url, protagonist_url]
    if panel_id in {'p02', 'p06', 'p07', 'p08'}:
        refs.append(mother_url)
    if prev_url and panel_id not in {'p01'}:
        refs = [prev_url] + refs

    print('Rendering', panel_id)
    if panel_id == 'p01':
        url = fal_generate(prompt)
    else:
        url = fal_edit(prompt, refs)
    raw_path = OUT / f'{panel_id}_raw.png'
    final_path = OUT / f'{panel_id}.png'
    download(url, raw_path)
    add_text_layer(raw_path, panel_id)
    raw_path.rename(final_path)
    prev_url = url
    rendered_paths.append((panel_id, final_path, panel['block_id']))
    manifest['panels'].append({
        'panel_id': panel_id,
        'url': url,
        'path': str(final_path.resolve()),
        'block_id': panel['block_id'],
        'shot': panel['shot'],
    })

spacing_px = {
    'tight': 30,
    'medium': 70,
    'tall_drop': 180,
    'end_cliff': 260,
}

images = []
total_h = 0
for i, (panel_id, path, block_id) in enumerate(rendered_paths):
    im = Image.open(path).convert('RGB')
    images.append((im, spacing_px.get(spacing_map.get(block_id, 'medium'), 70)))
    total_h += im.height
    if i < len(rendered_paths) - 1:
        total_h += spacing_px.get(spacing_map.get(block_id, 'medium'), 70)

canvas = Image.new('RGB', (PANEL_W, total_h), (244, 244, 244))
y = 0
for i, (im, gap) in enumerate(images):
    canvas.paste(im, (0, y))
    y += im.height
    if i < len(images) - 1:
        y += gap
longscroll = OUT / 'ep001_fal_longscroll.png'
canvas.save(longscroll)
manifest['longscroll'] = str(longscroll.resolve())
manifest['dimensions'] = {'panel': [PANEL_W, PANEL_H], 'longscroll': list(canvas.size)}
(BASE / 'generated_fal_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
