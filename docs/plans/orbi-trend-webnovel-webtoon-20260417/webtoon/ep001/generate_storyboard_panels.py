from PIL import Image, ImageDraw, ImageFilter
import math
import os

OUT_DIR = "/home/orbibot/.zeroclaw/workspace/hermes-agent/docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001/generated_panels"
W, H = 1080, 1600

PALETTE = {
    "bg": (18, 22, 30),
    "wall": (38, 44, 56),
    "wall2": (52, 58, 72),
    "desk": (106, 79, 56),
    "desk_dark": (74, 54, 39),
    "paper": (228, 224, 214),
    "paper_shadow": (190, 184, 171),
    "hoodie": (24, 27, 34),
    "skin": (227, 201, 182),
    "hair": (17, 18, 22),
    "mom": (167, 160, 151),
    "hall_light": (240, 242, 247),
    "laptop": (64, 76, 96),
    "glow": (130, 175, 230),
    "red": (187, 43, 43),
    "shadow": (8, 10, 14),
}


def gradient_bg(img, top, bottom):
    px = img.load()
    for y in range(img.height):
        t = y / max(1, img.height - 1)
        c = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        for x in range(img.width):
            px[x, y] = c


def add_noise(img, strength=12):
    px = img.load()
    for y in range(img.height):
        for x in range(img.width):
            n = ((x * 37 + y * 17 + (x * y) % 19) % strength) - strength // 2
            r, g, b = px[x, y]
            px[x, y] = (
                max(0, min(255, r + n)),
                max(0, min(255, g + n)),
                max(0, min(255, b + n)),
            )


def overlay_vignette(img, amt=140):
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for i in range(12):
        alpha = int(amt * (i + 1) / 12 * 0.12)
        d.rounded_rectangle((i * 30, i * 40, W - i * 30, H - i * 35), radius=40, outline=(0, 0, 0, alpha), width=60)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def draw_wall(draw, cracks=True):
    draw.rectangle((0, 0, W, H), fill=PALETTE["wall"])
    for y in range(0, H, 110):
        draw.line((0, y, W, y + 18), fill=PALETTE["wall2"], width=2)
    if cracks:
        crack_color = (70, 76, 86)
        paths = [
            [(140, 240), (170, 320), (150, 430), (190, 520)],
            [(860, 160), (845, 280), (900, 370), (880, 510)],
            [(540, 1000), (500, 1080), (560, 1200), (530, 1330)],
        ]
        for pts in paths:
            for a, b in zip(pts, pts[1:]):
                draw.line((*a, *b), fill=crack_color, width=3)
                draw.line((a[0] + 10, a[1] + 5, b[0] - 14, b[1] + 6), fill=crack_color, width=1)


def draw_desk(draw, bbox, clutter=True):
    x0, y0, x1, y1 = bbox
    draw.rounded_rectangle(bbox, radius=18, fill=PALETTE["desk"], outline=PALETTE["desk_dark"], width=8)
    draw.rectangle((x0 + 50, y0 + 45, x1 - 30, y1 - 40), outline=(135, 110, 88), width=3)
    if clutter:
        for ox, oy, w, h in [(85, 70, 210, 120), (320, 105, 150, 95), (520, 62, 170, 150), (730, 118, 180, 100)]:
            draw.rounded_rectangle((x0 + ox, y0 + oy, x0 + ox + w, y0 + oy + h), radius=10, fill=PALETTE["paper_shadow"], outline=(150, 144, 132))


def draw_paper(draw, bbox, tilt=0, marks=False):
    x0, y0, x1, y1 = bbox
    draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=PALETTE["paper"], outline=PALETTE["paper_shadow"], width=4)
    for y in range(y0 + 36, y1 - 20, 26):
        draw.line((x0 + 30, y, x1 - 30, y), fill=(175, 176, 180), width=2)
        for x in range(x0 + 60, x1 - 60, 180):
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(125, 127, 132))
    if marks:
        for cx, cy, r in [(x0 + 230, y0 + 210, 56), (x0 + 220, y0 + 215, 74), (x0 + 235, y0 + 220, 92)]:
            draw.arc((cx-r, cy-r, cx+r, cy+r), 25, 345, fill=PALETTE["red"], width=8)
            draw.arc((cx-r+8, cy-r+6, cx+r-14, cy+r-4), 15, 320, fill=PALETTE["red"], width=5)


def draw_hand(draw, x, y, scale=1.0, angle=0.0, tense=False):
    palm = (int(90 * scale), int(54 * scale))
    skin = PALETTE["skin"]
    draw.rounded_rectangle((x, y, x + palm[0], y + palm[1]), radius=int(18 * scale), fill=skin)
    finger_w = int(18 * scale)
    lengths = [78, 92, 88, 68]
    for i, ln in enumerate(lengths):
        fx = x + 10 + i * int(20 * scale)
        dy = -ln if not tense else -ln - (i % 2) * int(8 * scale)
        y0 = y + 6 + dy
        y1 = y + 6
        draw.rounded_rectangle((fx, min(y0, y1), fx + finger_w, max(y0, y1)), radius=int(8 * scale), fill=skin)
    draw.rounded_rectangle((x - int(16 * scale), y + int(20 * scale), x + int(18 * scale), y + int(58 * scale)), radius=int(10 * scale), fill=skin)


def draw_pen(draw, x, y, length=180, angle=-0.6):
    dx = math.cos(angle) * length
    dy = math.sin(angle) * length
    draw.line((x, y, x + dx, y + dy), fill=PALETTE["red"], width=14)
    draw.line((x + dx - 12, y + dy - 12, x + dx + 12, y + dy + 12), fill=(220, 220, 224), width=8)


def draw_student(draw, cx, cy, scale=1.0, facing="right", seated=True, glow=False):
    s = scale
    head_r = int(56 * s)
    skin = PALETTE["skin"]
    hair = PALETTE["hair"]
    hoodie = PALETTE["hoodie"]
    body_w, body_h = int(200 * s), int(260 * s)
    body_x0 = cx - body_w // 2
    body_y0 = cy
    draw.rounded_rectangle((body_x0, body_y0, body_x0 + body_w, body_y0 + body_h), radius=int(36 * s), fill=hoodie)
    draw.polygon([(body_x0 + 32, body_y0 + 10), (cx, body_y0 - 70), (body_x0 + body_w - 32, body_y0 + 10)], fill=(35, 38, 48))
    hx = cx + (25 if facing == "right" else -25) * s
    hy = cy - int(30 * s)
    draw.ellipse((hx - head_r, hy - head_r, hx + head_r, hy + head_r), fill=skin)
    draw.pieslice((hx - head_r - 8, hy - head_r - 22, hx + head_r + 10, hy + head_r - 8), 180, 360, fill=hair)
    draw.polygon([(hx - head_r + 8, hy - 6), (hx + head_r - 6, hy - 18), (hx + head_r - 10, hy + 8), (hx - head_r + 6, hy + 10)], fill=hair)
    eye_y = hy + 4
    if facing == "right":
        draw.line((hx + 8, eye_y, hx + 34, eye_y), fill=(20, 20, 22), width=4)
        draw.line((hx - 20, eye_y - 3, hx + 4, eye_y - 3), fill=(20, 20, 22), width=4)
        draw.rectangle((hx - 16, eye_y - 14, hx + 14, eye_y + 10), outline=(25, 25, 28), width=4)
        draw.rectangle((hx + 16, eye_y - 14, hx + 44, eye_y + 10), outline=(25, 25, 28), width=4)
        draw.line((hx + 12, eye_y - 2, hx + 18, eye_y - 2), fill=(25,25,28), width=3)
    else:
        draw.line((hx - 34, eye_y, hx - 8, eye_y), fill=(20, 20, 22), width=4)
        draw.line((hx - 4, eye_y - 3, hx + 20, eye_y - 3), fill=(20, 20, 22), width=4)
        draw.rectangle((hx - 44, eye_y - 14, hx - 16, eye_y + 10), outline=(25, 25, 28), width=4)
        draw.rectangle((hx - 14, eye_y - 14, hx + 16, eye_y + 10), outline=(25, 25, 28), width=4)
        draw.line((hx - 18, eye_y - 2, hx - 12, eye_y - 2), fill=(25,25,28), width=3)
    draw.line((hx + (12 if facing == "right" else -12), hy + 22, hx + (18 if facing == "right" else -18), hy + 42), fill=(180, 150, 140), width=3)
    if glow:
        draw.ellipse((hx - 130, hy - 80, hx + 150, hy + 210), outline=(140, 180, 225), width=6)


def draw_mother(draw, cx, cy, scale=1.0, facing="left", cold=True):
    s = scale
    head_r = int(54 * s)
    body_w, body_h = int(180 * s), int(280 * s)
    x0 = cx - body_w // 2
    y0 = cy
    draw.rounded_rectangle((x0, y0, x0 + body_w, y0 + body_h), radius=int(24 * s), fill=PALETTE["mom"])
    hx = cx + (-18 if facing == "left" else 18) * s
    hy = cy - int(26 * s)
    draw.ellipse((hx - head_r, hy - head_r, hx + head_r, hy + head_r), fill=(223, 202, 186))
    draw.pieslice((hx - head_r - 10, hy - head_r - 25, hx + head_r + 10, hy + head_r), 180, 360, fill=(60, 58, 62))
    brow_y = hy - 4
    if facing == "left":
        draw.line((hx - 28, brow_y, hx - 4, brow_y - 8), fill=(40, 38, 42), width=4)
        draw.line((hx + 4, brow_y - 8, hx + 24, brow_y - 12), fill=(40, 38, 42), width=4)
    else:
        draw.line((hx - 24, brow_y - 12, hx - 4, brow_y - 8), fill=(40, 38, 42), width=4)
        draw.line((hx + 4, brow_y - 8, hx + 28, brow_y), fill=(40, 38, 42), width=4)
    draw.line((hx - 14, hy + 30, hx + 14, hy + 32), fill=(120, 95, 96), width=3)


def draw_laptop(draw, bbox, open_ratio=0.68, screen_glow=True, content="grid", dark=False):
    x0, y0, x1, y1 = bbox
    base_h = 66
    draw.rounded_rectangle((x0, y1 - base_h, x1, y1), radius=12, fill=(80, 82, 88), outline=(40, 42, 48), width=4)
    draw.rounded_rectangle((x0 + 30, y0, x1 - 30, y1 - base_h + 24), radius=16, fill=(12, 17, 22) if dark else PALETTE["laptop"], outline=(35, 40, 48), width=6)
    sx0, sy0, sx1, sy1 = x0 + 52, y0 + 24, x1 - 52, y1 - base_h
    if screen_glow:
        fill = (18, 28, 42) if dark else (96, 124, 158)
        draw.rounded_rectangle((sx0, sy0, sx1, sy1), radius=10, fill=fill)
    if content == "grid":
        for i in range(6):
            x = sx0 + i * (sx1 - sx0) / 6
            draw.line((x, sy0 + 40, x, sy1 - 28), fill=(170, 205, 228), width=2)
        for i in range(7):
            y = sy0 + 28 + i * (sy1 - sy0 - 60) / 7
            draw.line((sx0 + 16, y, sx1 - 16, y), fill=(170, 205, 228), width=2)
        draw.rectangle((sx0 + 30, sy0 + 22, sx1 - 32, sy0 + 50), outline=(196, 219, 238), width=3)
    elif content == "tabs":
        tab_w = (sx1 - sx0 - 30) / 4
        for i in range(4):
            draw.rounded_rectangle((sx0 + 10 + i * tab_w, sy0 + 8, sx0 + 10 + (i + 1) * tab_w - 8, sy0 + 36), radius=8, fill=(70, 85, 104))
        for i in range(3):
            draw.rounded_rectangle((sx0 + 18, sy0 + 62 + i * 106, sx1 - 18, sy0 + 140 + i * 106), radius=16, fill=(45, 57, 76))
        draw.rounded_rectangle((sx1 - 210, sy1 - 120, sx1 - 24, sy1 - 30), radius=18, fill=(116, 146, 188))
    elif content == "files":
        for i in range(6):
            y = sy0 + 34 + i * 66
            fill = (70, 87, 104)
            if i == 3:
                fill = (142, 84, 84)
            draw.rounded_rectangle((sx0 + 18, y, sx1 - 18, y + 44), radius=12, fill=fill)
            draw.rectangle((sx0 + 36, y + 12, sx0 + 110, y + 30), fill=(190, 210, 228))
    elif content == "memo":
        draw.rounded_rectangle((sx0 + 18, sy0 + 22, sx1 - 18, sy1 - 22), radius=14, fill=(205, 214, 222))
        for i in range(8):
            y = sy0 + 52 + i * 56
            draw.line((sx0 + 52, y, sx1 - 52, y), fill=(130, 136, 144), width=3)
        draw.rounded_rectangle((sx0 + 44, sy0 + 200, sx1 - 44, sy0 + 278), radius=14, fill=(252, 235, 139), outline=(220, 190, 90), width=3)
    elif content == "chat":
        for i in range(3):
            draw.rounded_rectangle((sx0 + 40, sy0 + 70 + i * 110, sx1 - 160, sy0 + 128 + i * 110), radius=16, fill=(87, 110, 144))
        draw.rounded_rectangle((sx1 - 340, sy1 - 180, sx1 - 40, sy1 - 118), radius=18, fill=(205, 214, 222))
        draw.rounded_rectangle((sx1 - 430, sy1 - 90, sx1 - 70, sy1 - 24), radius=18, fill=(225, 231, 236))


def draw_room_light(draw, laptop_bbox, strength=180):
    x0, y0, x1, y1 = laptop_bbox
    cx = (x0 + x1) // 2
    cy = y0 + 40
    for i in range(10):
        alpha = int(strength * (10 - i) / 10)
        r = 120 + i * 48
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(130, 175, 230, alpha), width=18)


def make_base():
    img = Image.new("RGB", (W, H))
    gradient_bg(img, PALETTE["bg"], (44, 50, 66))
    draw = ImageDraw.Draw(img)
    draw_wall(draw)
    add_noise(img, 10)
    return img


def p01():
    img = make_base()
    d = ImageDraw.Draw(img)
    draw_desk(d, (60, 940, 1020, 1530), clutter=False)
    draw_paper(d, (130, 1020, 790, 1450), marks=True)
    draw_hand(d, 700, 1180, scale=1.5, tense=True)
    draw_pen(d, 835, 1215, length=240, angle=-1.0)
    for ox, oy in [(120, 960), (800, 980), (840, 1130)]:
        d.rounded_rectangle((ox, oy, ox + 180, oy + 120), radius=10, fill=PALETTE["paper_shadow"], outline=(150, 144, 132))
    img = overlay_vignette(img, 180)
    return img


def p02():
    img = make_base()
    d = ImageDraw.Draw(img)
    draw_desk(d, (250, 960, 980, 1510), clutter=True)
    draw_student(d, 640, 930, scale=1.45, facing="right")
    draw_mother(d, 360, 660, scale=1.35, facing="right")
    draw_laptop(d, (560, 990, 930, 1335), content="grid")
    d.rectangle((0, 0, W, 220), fill=(24, 28, 36))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    img = overlay_vignette(img, 150)
    return img


def p03():
    img = make_base()
    d = ImageDraw.Draw(img)
    draw_desk(d, (110, 980, 980, 1495), clutter=True)
    draw_laptop(d, (190, 520, 900, 1270), content="grid")
    for i in range(8):
        x = 210 + i * 78
        d.rectangle((x, 650, x + 42, 1160), outline=(190, 220, 234), width=2)
    d.ellipse((180, 1120, 980, 1600), fill=(26, 31, 40))
    d.rounded_rectangle((680, 1080, 820, 1320), radius=32, fill=(20, 24, 30))
    d.rounded_rectangle((820, 1080, 920, 1325), radius=26, fill=(20, 24, 30))
    d.ellipse((722, 1000, 894, 1165), fill=(28, 33, 40))
    d.ellipse((705, 980, 880, 1148), fill=(227, 201, 182))
    d.pieslice((695, 950, 890, 1090), 180, 360, fill=(17, 18, 22))
    img = overlay_vignette(img, 160)
    return img


def p04():
    img = Image.new("RGB", (W, H), (12, 15, 20))
    d = ImageDraw.Draw(img)
    draw_laptop(d, (90, 160, 990, 1420), content="tabs", dark=True)
    d.ellipse((560, 350, 950, 1040), fill=(26, 30, 38))
    d.ellipse((620, 420, 900, 830), fill=(225, 198, 180))
    d.pieslice((590, 330, 920, 760), 180, 360, fill=(16, 17, 22))
    d.rectangle((680, 580, 860, 670), outline=(28, 30, 34), width=5)
    d.rectangle((865, 580, 940, 670), outline=(28, 30, 34), width=5)
    for i in range(7):
        x = 120 + i * 140
        d.line((x, 160, x + 80, 1410), fill=(18, 22, 28), width=2)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    img = overlay_vignette(img, 120)
    return img


def p05():
    img = make_base()
    d = ImageDraw.Draw(img)
    draw_desk(d, (80, 980, 1020, 1520), clutter=True)
    draw_laptop(d, (170, 380, 940, 1220), content="files")
    d.rounded_rectangle((735, 1190, 930, 1300), radius=40, fill=(75, 80, 88), outline=(35, 38, 44), width=4)
    draw_hand(d, 640, 1140, scale=1.35, tense=True)
    for i in range(5):
        y = 1100 + i * 20
        d.line((620, y, 610 - i * 8, y + 25), fill=(215, 220, 230), width=3)
    img = overlay_vignette(img, 170)
    return img


def p06():
    img = make_base()
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, H), fill=(24, 28, 38))
    d.rectangle((0, 0, 220, H), fill=(236, 238, 244))
    d.rectangle((200, 0, 260, H), fill=(185, 173, 158))
    d.polygon([(220, 0), (220, H), (460, H - 160), (460, 140)], fill=(52, 58, 72))
    draw_desk(d, (500, 1030, 1030, 1540), clutter=False)
    draw_student(d, 790, 940, scale=1.35, facing="left")
    draw_laptop(d, (640, 880, 980, 1240), content="files")
    draw_mother(d, 150, 720, scale=1.5, facing="right")
    d.ellipse((50, 560, 260, 820), outline=(90, 92, 98), width=10)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.2))
    img = overlay_vignette(img, 160)
    return img


def p07():
    img = make_base()
    d = ImageDraw.Draw(img)
    draw_laptop(d, (150, 260, 930, 1360), content="memo")
    d.ellipse((40, 980, 500, 1570), fill=(40, 44, 52))
    d.ellipse((700, 940, 1080, 1600), fill=(90, 92, 98))
    img = img.filter(ImageFilter.GaussianBlur(radius=3.2))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((235, 675, 845, 755), radius=18, outline=(248, 224, 96), width=8)
    img = overlay_vignette(img, 130)
    return img


def p08():
    img = make_base()
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 220), fill=(16, 18, 24))
    draw_desk(d, (180, 1010, 930, 1510), clutter=True)
    draw_laptop(d, (330, 720, 780, 1180), content="chat")
    draw_student(d, 570, 1110, scale=1.25, facing="right", glow=True)
    draw_mother(d, 880, 760, scale=1.42, facing="left")
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for r, a in [(240, 40), (340, 28), (460, 18)]:
        gd.ellipse((550-r, 930-r, 550+r, 930+r), fill=(120, 175, 232, a))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((650, 380, 1000, 530), radius=24, fill=(220, 226, 232), outline=(90, 100, 120), width=4)
    d.rounded_rectangle((680, 565, 1040, 750), radius=28, fill=(235, 238, 242), outline=(90, 100, 120), width=4)
    img = overlay_vignette(img, 175)
    return img


def save_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    panels = [p01(), p02(), p03(), p04(), p05(), p06(), p07(), p08()]
    names = [f"ep001_{i:02d}_storyboard.png" for i in range(1, 9)]
    paths = []
    for img, name in zip(panels, names):
        path = os.path.join(OUT_DIR, name)
        img.save(path)
        paths.append(path)
    gap = 80
    long_h = len(panels) * H + (len(panels)-1) * gap
    scroll = Image.new("RGB", (W, long_h), (14, 16, 20))
    y = 0
    for img in panels:
        scroll.paste(img, (0, y))
        y += H + gap
    scroll_path = os.path.join(OUT_DIR, "ep001_storyboard_longscroll.png")
    scroll.save(scroll_path)
    return paths + [scroll_path]


if __name__ == "__main__":
    for p in save_all():
        print(p)
