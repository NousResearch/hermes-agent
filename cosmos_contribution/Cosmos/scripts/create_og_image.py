#!/usr/bin/env python3
"""Generate Open Graph image for Twitter/social media cards."""

from PIL import Image, ImageDraw, ImageFont
import random
import os

def create_og_image():
    # Create image 1200x630 (Twitter card size)
    width, height = 1200, 630
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # Create cosmic gradient background
    for y in range(height):
        r = int(15 + (y/height) * 20)
        g = int(10 + (y/height) * 15)
        b = int(40 + (y/height) * 60)
        for x in range(width):
            xvar = int((x/width) * 20)
            draw.point((x, y), fill=(r + xvar, g, b + 20 - xvar))

    # Add stars
    random.seed(42)
    for _ in range(200):
        x = random.randint(0, width)
        y = random.randint(0, height)
        brightness = random.randint(150, 255)
        size = random.randint(1, 3)
        draw.ellipse([x-size, y-size, x+size, y+size], fill=(brightness, brightness, brightness))

    # Add center glow effect
    for r in range(200, 0, -5):
        alpha = int(30 * (1 - r/200))
        color = (99 + alpha, 102 + alpha, 241)
        draw.ellipse([width//2-r, height//2-r-50, width//2+r, height//2+r-50], fill=color)

    # Brain glow (top center)
    brain_x, brain_y = width//2, 180
    for r in range(80, 0, -2):
        alpha = int(100 * (1 - r/80))
        draw.ellipse([brain_x-r, brain_y-r, brain_x+r, brain_y+r], fill=(150 + alpha, 100 + alpha, 255))

    # Load fonts
    try:
        title_font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 72)
        sub_font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 32)
        small_font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    except:
        try:
            title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 72)
            sub_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
            small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
        except:
            title_font = ImageFont.load_default()
            sub_font = title_font
            small_font = title_font

    # Draw brain emoji text
    try:
        emoji_font = ImageFont.truetype('C:/Windows/Fonts/seguiemj.ttf', 100)
        draw.text((width//2 - 50, 120), '🧠', font=emoji_font, fill=(255, 255, 255))
    except:
        pass

    # Title
    title = 'FARNSWORTH AI'
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text((width//2 - tw//2, 280), title, fill=(255, 255, 255), font=title_font)

    # Subtitle
    subtitle = 'Your Claude Companion with Superpowers'
    bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
    tw = bbox[2] - bbox[0]
    draw.text((width//2 - tw//2, 370), subtitle, fill=(200, 200, 220), font=sub_font)

    # Features
    features = 'Memory • Model Swarms • Solana Trading • P2P Network • Self-Evolution'
    bbox = draw.textbbox((0, 0), features, font=small_font)
    tw = bbox[2] - bbox[0]
    draw.text((width//2 - tw//2, 430), features, fill=(150, 150, 180), font=small_font)

    # URL
    url = 'ai.farnsworth.cloud'
    bbox = draw.textbbox((0, 0), url, font=sub_font)
    tw = bbox[2] - bbox[0]
    draw.text((width//2 - tw//2, 550), url, fill=(99, 102, 241), font=sub_font)

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'farnsworth', 'web', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'og-image.png')
    img.save(output_path, 'PNG')
    print(f'OG image created: {output_path}')
    return output_path

if __name__ == '__main__':
    create_og_image()
