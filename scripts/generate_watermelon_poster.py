from PIL import Image, ImageDraw, ImageFont, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pathlib import Path

DPI = 300
TRIM_W, TRIM_H = 9449, 11811
BLEED = 35
CANVAS_W, CANVAS_H = 9520, 11882
ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'outputs' / 'posters' / 'watermelon-firepalace-20260422'
OUTDIR.mkdir(parents=True, exist_ok=True)

USER_IMG = Path('/Users/market/.hermes/profiles/storemanager-lab/cache/images/img_ef730c55e383.jpg')
LOGO_SHEET = Path('/Users/market/Documents/GitHub/huazhuo-os/runtime-pack/workspaces/hzstoremangerbot/runtime-data/assets-hot/previews/large/fp-t3-brand-logo-guideline-001.png')

PNG_OUT = OUTDIR / 'firepalace-watermelon-poster-80x100cm-bleed-300dpi.png'
PDF_OUT = OUTDIR / 'firepalace-watermelon-poster-80x100cm-bleed-300dpi.pdf'
PREVIEW_OUT = OUTDIR / 'firepalace-watermelon-poster-preview.jpg'
LOGO_OUT = OUTDIR / 'firepalace-logo-block.png'

FONT_CN_BOLD = '/System/Library/Fonts/Hiragino Sans GB.ttc'
FONT_CN_ALT = '/System/Library/Fonts/STHeiti Medium.ttc'
FONT_EN = '/System/Library/Fonts/HelveticaNeue.ttc'


def mm_to_pt(mm: float) -> float:
    return mm / 25.4 * 72.0


def load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def make_vertical_gradient(size, top_rgb, bottom_rgb):
    w, h = size
    base = Image.new('RGB', size, top_rgb)
    draw = ImageDraw.Draw(base)
    for y in range(h):
        t = y / max(h - 1, 1)
        color = tuple(int(top_rgb[i] * (1 - t) + bottom_rgb[i] * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=color)
    return base


def make_radial_glow(size, center, radius, color, alpha_max):
    w, h = size
    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    px = layer.load()
    cx, cy = center
    r = radius
    cr, cg, cb = color
    for y in range(max(0, cy - r), min(h, cy + r)):
        for x in range(max(0, cx - r), min(w, cx + r)):
            dx = x - cx
            dy = y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= r:
                t = 1 - dist / r
                alpha = int(alpha_max * (t ** 2))
                px[x, y] = (cr, cg, cb, alpha)
    return layer.filter(ImageFilter.GaussianBlur(radius=48))


def fit_cover(img: Image.Image, target_size, focus=(0.5, 0.5)):
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


def rounded_mask(size, radius):
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask


def add_centered_text(draw, text, font, y, fill, canvas_w, tracking=0):
    if tracking == 0:
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        x = (canvas_w - width) // 2
        draw.text((x, y), text, font=font, fill=fill)
        return x, y, width, bbox[3] - bbox[1]
    chars = list(text)
    widths = [draw.textbbox((0, 0), c, font=font)[2] for c in chars]
    total_w = sum(widths) + tracking * (len(chars) - 1)
    x = (canvas_w - total_w) // 2
    cursor = x
    for c, cw in zip(chars, widths):
        draw.text((cursor, y), c, font=font, fill=fill)
        cursor += cw + tracking
    height = draw.textbbox((0, 0), text, font=font)[3]
    return x, y, total_w, height


def draw_watermelon_badge(base: Image.Image, box):
    x1, y1, x2, y2 = box
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.pieslice(box, start=200, end=340, fill=(216, 54, 61, 230), outline=(29, 137, 74, 255), width=26)
    d.arc((x1 + 20, y1 + 20, x2 - 20, y2 - 20), start=200, end=340, fill=(243, 243, 228, 255), width=18)
    seed_positions = [(0.28, 0.44), (0.48, 0.34), (0.63, 0.48), (0.42, 0.58)]
    for sx, sy in seed_positions:
        cx = x1 + int((x2 - x1) * sx)
        cy = y1 + int((y2 - y1) * sy)
        d.ellipse((cx - 10, cy - 18, cx + 10, cy + 18), fill=(42, 24, 20, 220))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(0.4)))


def build():
    poster = make_vertical_gradient((CANVAS_W, CANVAS_H), (248, 246, 236), (234, 247, 231)).convert('RGBA')
    poster.alpha_composite(make_radial_glow((CANVAS_W, CANVAS_H), (CANVAS_W // 2, 1950), 2700, (255, 224, 180), 150))
    poster.alpha_composite(make_radial_glow((CANVAS_W, CANVAS_H), (CANVAS_W // 2, 7600), 3400, (255, 105, 105), 80))
    poster.alpha_composite(make_radial_glow((CANVAS_W, CANVAS_H), (1650, 9800), 2200, (120, 191, 110), 65))

    draw = ImageDraw.Draw(poster)

    logo = crop_logo_block(LOGO_SHEET)
    logo = fit_contain(logo, (2500, 850))
    logo_x = (CANVAS_W - logo.width) // 2
    logo_y = 610
    poster.alpha_composite(logo, (logo_x, logo_y))

    top_line_font = load_font(FONT_EN, 120)
    title_font = load_font(FONT_CN_BOLD, 440)
    subtitle_font = load_font(FONT_CN_ALT, 132)
    body_font = load_font(FONT_CN_ALT, 110)
    footer_font = load_font(FONT_CN_ALT, 96)

    add_centered_text(draw, 'WATERMELON JUICE', top_line_font, 1510, '#8C2323', CANVAS_W, tracking=8)
    add_centered_text(draw, '鲜榨西瓜汁', title_font, 1690, '#B11E25', CANVAS_W, tracking=4)
    add_centered_text(draw, '清爽鲜甜  ·  一口入夏', subtitle_font, 2210, '#547645', CANVAS_W, tracking=12)

    photo = Image.open(USER_IMG).convert('RGBA')
    photo = fit_cover(photo, (3980, 7320), focus=(0.5, 0.46))
    mask = rounded_mask(photo.size, 170)
    photo.putalpha(mask)

    shadow = Image.new('RGBA', (photo.width + 180, photo.height + 180), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle((60, 60, shadow.width - 60, shadow.height - 60), radius=205, fill=(111, 37, 37, 72))
    shadow = shadow.filter(ImageFilter.GaussianBlur(42))

    px = (CANVAS_W - photo.width) // 2
    py = 2720
    poster.alpha_composite(shadow, (px - 85, py + 55))
    poster.alpha_composite(photo, (px, py))

    deco = Image.new('RGBA', poster.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(deco)
    d.ellipse((1080, 3890, 1510, 4320), fill=(255, 255, 255, 105))
    d.ellipse((7990, 7440, 8420, 7870), fill=(255, 255, 255, 92))
    d.rounded_rectangle((1430, 10100, 8090, 10710), radius=130, fill=(255, 245, 236, 144))
    poster.alpha_composite(deco.filter(ImageFilter.GaussianBlur(16)))

    draw_watermelon_badge(poster, (1220, 2450, 1760, 2990))
    draw_watermelon_badge(poster, (7700, 9420, 8260, 9980))

    d2 = ImageDraw.Draw(poster)
    add_centered_text(d2, '甄选西瓜鲜榨  ·  色泽饱满  ·  入口沁凉', body_font, 10185, '#8F2A2A', CANVAS_W, tracking=8)
    add_centered_text(d2, '北京首都机场 T3 火宫殿', body_font, 10370, '#2E2E2E', CANVAS_W, tracking=5)
    add_centered_text(d2, '夏日清爽推荐', footer_font, 10615, '#547645', CANVAS_W, tracking=8)

    poster = poster.convert('RGB')
    poster.save(PNG_OUT, dpi=(DPI, DPI), quality=95)

    preview = poster.copy()
    preview.thumbnail((1600, 1995), Image.Resampling.LANCZOS)
    preview.save(PREVIEW_OUT, quality=92)

    c = canvas.Canvas(str(PDF_OUT), pagesize=(mm_to_pt(806), mm_to_pt(1006)))
    c.drawImage(ImageReader(str(PNG_OUT)), 0, 0, width=mm_to_pt(806), height=mm_to_pt(1006), preserveAspectRatio=False, mask='auto')
    c.setTitle('Fire Palace Watermelon Juice Poster 80x100cm')
    c.save()

    print('saved_png', PNG_OUT)
    print('saved_pdf', PDF_OUT)
    print('saved_preview', PREVIEW_OUT)
    print('saved_logo', LOGO_OUT)


if __name__ == '__main__':
    build()
