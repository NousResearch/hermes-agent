from __future__ import annotations

import textwrap
from pathlib import Path

import yaml
from PIL import Image, ImageColor, ImageDraw, ImageFont


BASE = Path("docs/plans/orbi-romance-webtoon-20260421")
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


TITLE_FONT = load_font(42)
BODY_FONT = load_font(30)
SMALL_FONT = load_font(24)
TINY_FONT = load_font(20)


SPACING_MAP = {
    "tight": 40,
    "medium": 80,
    "tall_drop": 170,
    "end_cliff": 250,
}


def wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def draw_gradient(img: Image.Image, top_hex: str, bottom_hex: str) -> None:
    top = ImageColor.getrgb(top_hex)
    bottom = ImageColor.getrgb(bottom_hex)
    px = img.load()
    for y in range(img.height):
        t = y / max(1, img.height - 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        for x in range(img.width):
            px[x, y] = color


def draw_figure(draw: ImageDraw.ImageDraw, center_x: int, base_y: int, accent: tuple[int, int, int], facing: int) -> None:
    head_r = 54
    head_x = center_x + 26 * facing
    head_y = base_y - 310
    draw.ellipse((head_x - head_r, head_y - head_r, head_x + head_r, head_y + head_r), fill=(233, 213, 198))
    draw.pieslice((head_x - head_r - 10, head_y - head_r - 28, head_x + head_r + 10, head_y + head_r), 180, 360, fill=(24, 25, 30))
    body = (center_x - 110, base_y - 240, center_x + 110, base_y)
    draw.rounded_rectangle(body, radius=40, fill=accent)
    draw.polygon([(center_x - 70, base_y - 210), (center_x, base_y - 300), (center_x + 70, base_y - 210)], fill=(44, 46, 52))


def draw_panel(
    panel_path: Path,
    panel: dict,
    block: dict,
    lettering: dict,
    queue: dict,
    episode_title: str,
    panel_index: int,
) -> None:
    width = int(queue["panel_size"]["width"])
    height = int(queue["panel_size"]["height"])
    palette = queue["palette"]
    bubble_color = ImageColor.getrgb(palette["bubble"])
    accent = ImageColor.getrgb(palette["accent"])
    ink = ImageColor.getrgb(palette["ink"])

    img = Image.new("RGB", (width, height))
    draw_gradient(img, palette["top_bg"], palette["bottom_bg"])
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((36, 34, width - 36, height - 34), radius=36, outline=(255, 255, 255), width=3)
    draw.rounded_rectangle((56, 56, width - 56, 160), radius=26, fill=(255, 255, 255, 60))
    draw.text((88, 78), f"{episode_title}  |  {panel['panel_id'].upper()}", font=TITLE_FONT, fill=(250, 250, 248))
    draw.text((88, 126), f"{block['emotion']} / {panel['shot']}", font=SMALL_FONT, fill=(235, 235, 230))

    draw.rounded_rectangle((70, 220, width - 70, height - 430), radius=32, outline=accent, width=8)
    if panel_index % 2 == 0:
        draw_figure(draw, 330, height - 470, (42, 54, 78), 1)
        draw_figure(draw, 760, height - 470, accent, -1)
    else:
        draw_figure(draw, 420, height - 470, accent, 1)
    draw.rounded_rectangle((180, 320, width - 180, 430), radius=22, fill=(255, 255, 255, 42))
    draw.text((210, 345), panel["prompt"][:90], font=SMALL_FONT, fill=(250, 250, 244))

    desc_box = (84, height - 390, width - 84, height - 170)
    draw.rounded_rectangle(desc_box, radius=24, fill=(250, 244, 232))
    draw.text((110, height - 362), wrap_text(block["visual"], 29), font=BODY_FONT, fill=ink, spacing=10)

    captions = [c["text"] for c in lettering.get("captions", []) if c["panel_id"] == panel["panel_id"]]
    balloons = [b["text"] for b in lettering.get("balloons", []) if b["panel_id"] == panel["panel_id"]]

    if captions:
        draw.rounded_rectangle((84, 184, 540, 294), radius=20, fill=(24, 28, 38))
        draw.text((112, 208), wrap_text(captions[0], 18), font=SMALL_FONT, fill=(255, 255, 255), spacing=8)

    bubble_y = height - 162
    for idx, text in enumerate(balloons[:2]):
        left = 92 + idx * 440
        right = min(left + 390, width - 92)
        draw.rounded_rectangle((left, bubble_y - 10, right, height - 72), radius=28, fill=bubble_color, outline=ink, width=3)
        draw.text((left + 24, bubble_y + 12), wrap_text(text, 11), font=SMALL_FONT, fill=ink, spacing=6)

    draw.text((width - 214, height - 66), panel["palette_tag"], font=TINY_FONT, fill=(240, 240, 238))
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(panel_path)


def render_episode(ep_dir: Path) -> Path:
    scroll = yaml.safe_load((ep_dir / "scroll_plan.yaml").read_text(encoding="utf-8"))
    prompts = yaml.safe_load((ep_dir / "panel_prompts.yaml").read_text(encoding="utf-8"))
    lettering = yaml.safe_load((ep_dir / "lettering_script.yaml").read_text(encoding="utf-8"))
    queue = yaml.safe_load((ep_dir / "render_queue.yaml").read_text(encoding="utf-8"))

    block_map = {block["block_id"]: block for block in scroll["blocks"]}
    output_dir = Path(queue["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[tuple[Image.Image, int]] = []
    for index, panel in enumerate(prompts["panels"], start=1):
        block = block_map[panel["block_id"]]
        panel_path = output_dir / f"{panel['panel_id']}.png"
        draw_panel(panel_path, panel, block, lettering, queue, scroll["source_title"], index)
        rendered.append((Image.open(panel_path).convert("RGB"), SPACING_MAP.get(block["spacing"], 80)))

    width = int(queue["panel_size"]["width"])
    total_height = 0
    for idx, (panel_img, gap) in enumerate(rendered):
        total_height += panel_img.height
        if idx < len(rendered) - 1:
            total_height += gap

    canvas = Image.new("RGB", (width, total_height), ImageColor.getrgb(queue["palette"]["bubble"]))
    y = 0
    for idx, (panel_img, gap) in enumerate(rendered):
        canvas.paste(panel_img, (0, y))
        y += panel_img.height
        if idx < len(rendered) - 1:
            y += gap

    longscroll = output_dir / queue["longscroll_name"]
    canvas.save(longscroll)
    return longscroll


def main() -> None:
    for ep_dir in sorted((BASE / "webtoon").glob("ep0*")):
        longscroll = render_episode(ep_dir)
        print(longscroll)


if __name__ == "__main__":
    main()
