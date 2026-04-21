from __future__ import annotations
import json
from pathlib import Path
import requests
import fal_client

BASE = Path('docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001')
OUT = BASE / 'generated_fal_v3_p02_candidates'
OUT.mkdir(parents=True, exist_ok=True)
PANEL_W = 720
PANEL_H = 1072

def fal_generate(prompt: str):
    return fal_client.subscribe(
        'fal-ai/flux-2-pro',
        arguments={
            'prompt': prompt,
            'image_size': {'width': PANEL_W, 'height': PANEL_H},
            'num_images': 1,
            'output_format': 'png',
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
            'output_format': 'png',
        },
    )['images'][0]['url']

def download(url: str, path: Path):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)

anchors = json.loads((BASE / 'generated_fal_manifest_v3.json').read_text(encoding='utf-8'))['anchors']
style = ('clean Korean webtoon cartoon illustration, polished digital manhwa style, 2D cel shading, expressive line art, '
         'soft cinematic lighting, consistent same apartment study room, same 19-year-old Korean male student in black hoodie, '
         'same Korean mother in simple beige homewear when present, emotionally readable faces, '
         'absolutely no text, no letters, no Korean characters, no English characters, no numbers, no captions, '
         'no speech balloons, no dialogue bubbles, no chat bubbles, no UI labels, no watermarks, no subtitles, '
         'no readable monitor text, no gibberish text, no document typography, no signs, no logos')

prompts = {
    'cand1_generate': style + ', over-the-shoulder shot from behind a Korean mother standing over seated male student in a cramped exam-prep bedroom, pressure-filled silence, both faces visible, clean wall and curtain shapes only, no bubble-shaped white objects anywhere, no oval white shapes, no comic balloons, no empty callout areas',
    'cand2_edit_anchors': style + ', same room and same characters, mother looming over seated son at desk, tense silent confrontation, keep top half of frame visually empty except room background, absolutely no white oval objects, no speech balloons, no bubble silhouettes, no text marks',
    'cand3_generate_wide': style + ', wider over-shoulder composition of mother and son in small Korean study room at night, desk glow, bookshelf, curtain, window, pressure-filled silence, all surfaces text-free, no bubble-like objects, no white floating shapes in upper area',
}
results = []
for name, prompt in prompts.items():
    if name == 'cand2_edit_anchors':
        url = fal_edit(prompt, [anchors['location'], anchors['protagonist'], anchors['mother']])
    else:
        url = fal_generate(prompt)
    path = OUT / f'{name}.png'
    download(url, path)
    results.append({'name': name, 'url': url, 'path': str(path.resolve()), 'prompt': prompt})

(OUT / 'manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, ensure_ascii=False, indent=2))
