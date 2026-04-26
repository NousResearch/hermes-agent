from PIL import Image, ImageDraw, ImageFont, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pathlib import Path

DPI = 300
BLEED_MM = 3
TRIM_MM = (800, 1000)
CANVAS_MM = (806, 1006)
TRIM_W, TRIM_H = 9449, 11811
BLEED = 35
CANVAS_W, CANVAS_H = 9520, 11882
ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'outputs' / 'posters' / 'watermelon-firepalace-20260422-v2'
OUTDIR.mkdir(parents=True, exist_ok=True)

USER_IMG = Path('/Users/market/.hermes/profiles/storemanager-lab/cache/images/img_fc1298e9650f.jpg')
LOGO_SHEET = Path('/Users/market/Documents/GitHub/huazhuo-os/runtime-pack/workspaces/hzstoremangerbot/runtime-data/assets-hot/previews/large/fp-t3-brand-logo-guideline-001.png')

PNG_OUT = OUTDIR / 'firepalace-watermelon-poster-fullbleed-80x100cm-300dpi.png'
PDF_OUT = OUTDIR / 'firepalace-watermelon-poster-fullbleed-80x100cm-300dpi.pdf'
PREVIEW_OUT = OUTDIR / 'firepalace-watermelon-poster-fullbleed-preview.jpg'
LOGO_OUT = OUTDIR / 'firepalace-logo-block.png'

FONT_CN_BOLD = '/System/Library/Fonts/Hiragino Sans GB.ttc'
FONT_CN = '/System/Library/Fonts/STHeiti Medium.ttc'
FONT_EN = '/System/Library/Fonts/HelveticaNeue.ttc'


def mm_to_pt(mm: float) -> float:
    return mm / 25.4 * 72.0


def load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def fit_cover(img: Image.Image, target_size, focus=(0.5, 0.44)):
    tw, th = target_size
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    resized = img.resize((int(iw * scale), int(ih * scale)), Image.Resampling.LANCZOS)
    rw, rh = resized.size
    fx, fy = focus
    left = max(0, min(int((rw - tw) * fx), rw - tw))
    top = max(0, min(int((rh - th) * fy), rh - th))
    return resized.crop((left, top, left + tw, top + th))


def fit_contain(img: Image.Image, target_size):
    tw, th = target_size
    iw, ih = img.size
    scale = min(tw / iw, th / ih)
    return img.resize((int(iw * scale), int(ih * scale)), Image.Resampling.LANCZOS)


def transparentize_near_white(img: Image.Image, threshold=246):
    img = img.convert('RGBA')
    px = img.load()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = px[x, y]
            if r >= threshold and g >= threshold and b >= threshold:
                px[x, y] = (255, 255, 255, 0)
    return img


def crop_logo_block(sheet: Path) -> Image.Image:
    img = Image.open(sheet).convert('RGBA')
    w, h = img.size
    first_block = img.crop((70, 35, w - 70, int(h * 0.26)))
    first_block = transparentize_near_white(first_block)
    bbox = first_block.getbbox()
    if bbox:
        cropped = first_block.crop((max(bbox[0] - 16, 0), max(bbox[1] - 16, 0), min(bbox[2] + 16, first_block.width), min(bbox[3] + 16, first_block.height)))
    else:
        cropped = first_block
    cropped.save(LOGO_OUT)
    return cropped


def add_centered_text(draw, text, font, y, fill, canvas_w, tracking=0, stroke_fill=None, stroke_width=0):
    if tracking == 0:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        width = bbox[2] - bbox[0]
        x = (canvas_w - width) // 2
        draw.text((x, y), text, font=font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_width)
        return x, y, width, bbox[3] - bbox[1]
    chars = list(text)
    widths = [draw.textbbox((0, 0), c, font=font, stroke_width=stroke_width)[2] for c in chars]
    total_w = sum(widths) + tracking * (len(chars) - 1)
    x = (canvas_w - total_w) // 2
    cursor = x
    for c, cw in zip(chars, widths):
        draw.text((cursor, y), c, font=font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_width)
        cursor += cw + tracking
    height = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)[3]
    return x, y, total_w, height


def make_gradient_overlay(size, top_alpha=160, bottom_alpha=210):
    w, h = size
    overlay = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(h):
        if y < h * 0.32:
            alpha = int(top_alpha * (1 - y / (h * 0.32)))
        elif y > h * 0.72:
            alpha = int(bottom_alpha * ((y - h * 0.72) / (h * 0.28)))
        else:
            alpha = 0
        draw.line((0, y, w, y), fill=(15, 30, 18, alpha))
    return overlay


def build():
    base = Image.open(USER_IMG).convert('RGB')
    poster = fit_cover(base, (CANVAS_W, CANVAS_H), focus=(0.5, 0.45)).convert('RGBA')

    soft = poster.filter(ImageFilter.GaussianBlur(24)).convert('RGBA')
    soft.putalpha(68)
    poster.alpha_composite(soft)
    poster.alpha_composite(make_gradient_overlay((CANVAS_W, CANVAS_H), 175, 220))

    highlight = Image.new('RGBA', (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    hd = ImageDraw.Draw(highlight)
    hd.ellipse((1100, 560, 8200, 3300), fill=(255, 255, 255, 36))
    hd.ellipse((1400, 7600, 8200, 11200), fill=(255, 255, 255, 18))
    poster.alpha_composite(highlight.filter(ImageFilter.GaussianBlur(110)))

    draw = ImageDraw.Draw(poster)
    logo = crop_logo_block(LOGO_SHEET)
    logo = fit_contain(logo, (2500, 880))
    lx = (CANVAS_W - logo.width) // 2
    ly = 520
    poster.alpha_composite(logo, (lx, ly))

    font_en = load_font(FONT_EN, 112)
    font_title = load_font(FONT_CN_BOLD, 470)
    font_sub = load_font(FONT_CN, 128)
    font_badge = load_font(FONT_CN, 104)
    font_small = load_font(FONT_CN, 86)

    badge_box = (620, 1520, 2320, 1990)
    panel = Image.new('RGBA', (badge_box[2]-badge_box[0], badge_box[3]-badge_box[1]), (255,255,255,0))
    pd = ImageDraw.Draw(panel)
    pd.rounded_rectangle((0,0,panel.width,panel.height), radius=70, fill=(255,248,236,190), outline=(255,255,255,120), width=4)
    panel = panel.filter(ImageFilter.GaussianBlur(0.2))
    poster.alpha_composite(panel, (badge_box[0], badge_box[1]))
    draw = ImageDraw.Draw(poster)
    draw.text((760, 1625), '夏日鲜饮推荐', font=font_badge, fill='#8B1E25')

    add_centered_text(draw, 'FRESH WATERMELON JUICE', font_en, 2100, '#FFF6ED', CANVAS_W, tracking=8)
    add_centered_text(draw, '鲜榨西瓜汁', font_title, 2280, '#FFF6ED', CANVAS_W, tracking=4, stroke_fill='#8B1E25', stroke_width=2)
    add_centered_text(draw, '清甜现打  ·  冰爽解暑', font_sub, 2820, '#E7F6D9', CANVAS_W, tracking=10)

    bottom_panel = Image.new('RGBA', (CANVAS_W - 980, 1320), (0,0,0,0))
    bd = ImageDraw.Draw(bottom_panel)
    bd.rounded_rectangle((0, 0, bottom_panel.width, bottom_panel.height), radius=110, fill=(255, 248, 240, 172), outline=(255,255,255,95), width=3)
    bottom_panel = bottom_panel.filter(ImageFilter.GaussianBlur(0.2))
    poster.alpha_composite(bottom_panel, (490, 9930))
    draw = ImageDraw.Draw(poster)

    add_centered_text(draw, '甄选西瓜鲜榨  ·  色泽清透  ·  入口沁凉', font_badge, 10170, '#7E1B21', CANVAS_W, tracking=6)
    add_centered_text(draw, '北京首都机场 T3 火宫殿', font_badge, 10370, '#2C2C2C', CANVAS_W, tracking=4)
    add_centered_text(draw, '印刷规格：80 × 100 cm  |  300 DPI  |  四边各留 3 mm 出血', font_small, 10630, '#4C5E44', CANVAS_W, tracking=2)

    poster = poster.convert('RGB')
    poster.save(PNG_OUT, dpi=(DPI, DPI), quality=95)

    preview = poster.copy()
    preview.thumbnail((1600, 1995), Image.Resampling.LANCZOS)
    preview.save(PREVIEW_OUT, quality=92)

    c = canvas.Canvas(str(PDF_OUT), pagesize=(mm_to_pt(CANVAS_MM[0]), mm_to_pt(CANVAS_MM[1])))
    c.drawImage(ImageReader(str(PNG_OUT)), 0, 0, width=mm_to_pt(CANVAS_MM[0]), height=mm_to_pt(CANVAS_MM[1]), preserveAspectRatio=False, mask='auto')
    c.setTitle('Fire Palace Watermelon Juice Poster Fullbleed 80x100cm')
    c.save()

    print('saved_png', PNG_OUT)
    print('saved_pdf', PDF_OUT)
    print('saved_preview', PREVIEW_OUT)
    print('saved_logo', LOGO_OUT)


if __name__ == '__main__':
    build()
