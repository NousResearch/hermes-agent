#!/usr/bin/env python3
"""Generate voice-ready palette descriptions for Meilisearch indexing.

Reads brand.json files from palettes/flagship/, produces natural-language
descriptions that language models can reason over for voice queries.

Usage:
    python3 generate_descriptions.py > descriptions.json
    python3 generate_descriptions.py --single hugos-mom-1974
"""

import json
import os
import sys
import colorsys


def hex_to_hsl(hex_color):
    """Convert #RRGGBB to (H, S, L) in 0-360, 0-100, 0-100."""
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s * 100, l * 100


def describe_hue(h, s, l):
    """Describe a color in natural, evocative language from HSL values.
    
    Returns descriptions like 'warm salmon-coral' or 'dusty mauve' 
    rather than 'soft medium-light red' — optimized for voice queryability.
    """
    # Achromatic — no discernible hue
    if s < 10:
        if l < 15:
            return "near-black"
        if l < 30:
            return "dark charcoal"
        if l < 50:
            return "warm gray"
        if l < 70:
            return "silver"
        if l < 85:
            return "pale gray"
        return "ivory"

    # Evocative hue names — these are what make the descriptions voice-queryable
    # Instead of "orange-red", someone might say "terracotta" or "coral"
    hue_names = [
        (10,  "scarlet"),
        (20,  "coral"),
        (35,  "terracotta"),
        (45,  "burnt orange"),
        (55,  "amber"),
        (70,  "gold"),
        (90,  "chartreuse-yellow"),
        (110, "olive"),
        (140, "forest green"),
        (165, "emerald"),
        (180, "teal"),
        (200, "cerulean"),
        (220, "steel blue"),
        (245, "cobalt"),
        (265, "indigo"),
        (285, "violet"),
        (310, "mauve"),
        (330, "dusty rose"),
        (345, "burgundy"),
        (360, "scarlet"),
    ]

    hue_name = "scarlet"
    for threshold, name in hue_names:
        if h <= threshold:
            hue_name = name
            break

    # Muted/saturated qualifiers that add color character
    if s < 20:
        prefix = "dusty" if l < 60 else "pale"
    elif s < 40:
        prefix = "muted" if l < 50 else "soft"
    elif s < 60:
        prefix = ""  # no qualifier — just the hue name
    elif s < 80:
        prefix = "rich" if l < 50 else "bright"
    else:
        prefix = "vivid" if l < 50 else "electric"

    # Lightness qualifiers — skip for mid-tones, add for extremes
    if l < 20:
        lightness = "deep "
    elif l < 35:
        lightness = "dark "
    elif l > 80:
        lightness = "pale "
    elif l > 65:
        lightness = "light "
    else:
        lightness = ""  # mid-tone — no qualifier needed

    # Combine — avoid stacking more than one qualifier before the hue name
    # "deep coral" is good, "electric light dusty rose" is bad
    # Priority: saturation prefix wins over lightness qualifier for mid-tones
    if prefix and lightness:
        # Both want to qualify — pick the more evocative one
        # For deep/dark colors, lightness qualifier is more informative
        # For pale/light colors, lightness qualifier is more informative
        # For mid-tones, saturation qualifier is more informative
        if lightness in ("deep ", "dark "):
            parts = [lightness.strip(), hue_name]
        elif lightness in ("pale ", "light "):
            parts = [lightness.strip(), hue_name]
        else:
            parts = [prefix, hue_name]
    elif prefix:
        parts = [prefix, hue_name]
    elif lightness:
        parts = [lightness.strip(), hue_name]
    else:
        parts = [hue_name]
    return " ".join(parts)


def describe_palette_character(highlight_hsl, support_hsl, pv, is_dark):
    """Describe the overall palette character in 2-3 words."""
    h_h, h_s, h_l = highlight_hsl
    s_h, s_s, s_l = support_hsl
    
    # Warmth based on hue of Highlight and Support
    warm = 0
    if 0 < h_h < 70 or h_h > 310:
        warm += 1
    if 0 < s_h < 70 or s_h > 310:
        warm += 0.5
    if 140 < h_h < 310:
        warm -= 1
    if 140 < s_h < 310:
        warm -= 0.5

    # Intensity from perceptual volume
    if pv > 0.015:
        intensity = "bold"
    elif pv > 0.008:
        intensity = "cinematic"
    elif pv > 0.004:
        intensity = "rich"
    elif pv > 0.002:
        intensity = "refined"
    else:
        intensity = "minimal"

    # Mood based on warmth + darkness
    if warm > 0.5:
        temp = "warm"
    elif warm < -0.5:
        temp = "cool"
    else:
        temp = "neutral"

    if is_dark:
        if warm > 0.5:
            depth = "cinematic"
        elif warm < -0.5:
            depth = "dramatic"
        else:
            depth = "intimate" if pv < 0.005 else "sophisticated"
    else:
        depth = "airy" if pv < 0.005 else "vibrant"

    # Build final mood: avoid duplicating the intensity word
    moods = [temp]
    if depth != intensity:  # avoid "cinematic, cinematic"
        moods.append(depth)
    moods.append(intensity)
    return ", ".join(moods)


def pv_label(pv):
    """Describe perceptual volume in natural language."""
    if pv < 0.002:
        return "minimal color variety — nearly monochrome"
    elif pv < 0.005:
        return "limited palette — focused and restrained"
    elif pv < 0.010:
        return "moderate color range — versatile and balanced"
    elif pv < 0.020:
        return "rich color range — expressive and dynamic"
    else:
        return "vibrant color diversity — bold and diverse"


def contrast_note(text_bg, highlight_bg, canvas_bg):
    """Describe contrast accessibility in natural language."""
    notes = []
    if text_bg < 3:
        notes.append(f"body text has low contrast on the background ({text_bg:.1f}:1) — use Canvas as text surface instead")
    elif text_bg < 4.5:
        notes.append(f"body text is marginally readable on the background ({text_bg:.1f}:1)")
    else:
        notes.append(f"body text reads clearly on the background ({text_bg:.1f}:1)")
    
    if highlight_bg >= 4.5:
        notes.append(f"headings pass WCAG AA at {highlight_bg:.1f}:1")
    elif highlight_bg >= 3:
        notes.append(f"headings pass WCAG AA for large text at {highlight_bg:.1f}:1")
    else:
        notes.append(f"headings have marginal contrast at {highlight_bg:.1f}:1 — consider a lighter Highlight tint")
    
    return "; ".join(notes)


def generate_description(brand_data, theme_data):
    """Generate a voice-ready description for a single theme."""
    roles = brand_data['roles']
    contrast = brand_data.get('contrast', {})
    
    # Get HSL for key roles
    bg_hsl = hex_to_hsl(roles['Background']['hex'])
    hl_hsl = hex_to_hsl(roles['Highlight']['hex'])
    sup_hsl = hex_to_hsl(roles['Support']['hex'])
    c1_hsl = hex_to_hsl(roles['Chart1']['hex'])
    
    pv = theme_data.get('perceptual_volume', 0)
    is_dark = bg_hsl[2] < 50
    
    year = brand_data.get('year', '')
    source = brand_data.get('source', theme_data.get('name', ''))
    source_type = brand_data.get('source_type', '')
    source_desc = f"{source} ({source_type})" if source_type else source
    
    hl_desc = describe_hue(*hl_hsl)
    sup_desc = describe_hue(*sup_hsl)
    canvas_desc = describe_hue(*hex_to_hsl(roles['Canvas']['hex']))
    character = describe_palette_character(hl_hsl, sup_hsl, pv, is_dark)
    
    text_bg = contrast.get('text_on_background', 0)
    highlight_bg = contrast.get('highlight_on_background', 0)
    canvas_bg = contrast.get('canvas_on_background', 0)
    contrast_text = contrast_note(text_bg, highlight_bg, canvas_bg)
    
    # Best-for based on character
    best_for = []
    if pv > 0.01:
        best_for.append("bold creative presentations")
    elif pv > 0.005:
        best_for.append("cinematic presentations")
    else:
        best_for.append("elegant focused presentations")
    
    if is_dark:
        best_for.append("dark mode dashboards")
    if hl_hsl[1] > 50:  # saturated highlight
        best_for.append("attention-grabbing CTAs")
    if pv < 0.005:
        best_for.append("editorial layouts")
    if "warm" in character:
        best_for.append("photography portfolios")
    
    mood = "dark" if is_dark else "light"
    
    desc = (
        f'"{brand_data["name"]}" ({year}) — extracted from {source_desc}. '
        f'A {character} {mood} palette. The Highlight is {hl_desc} ({roles["Highlight"]["hex"]}), '
        f'paired with {sup_desc} Support ({roles["Support"]["hex"]}). '
        f'{"Near-black" if is_dark else "Light"} background ({roles["Background"]["hex"]}) '
        f'with {canvas_desc} Canvas ({roles["Canvas"]["hex"]}). '
        f'{contrast_text}. '
        f'{pv_label(pv)}. '
        f'Best for {", ".join(best_for)}.'
    )
    
    return desc


if __name__ == '__main__':
    # All data is in the brand JSONs — no external dependency needed
    # Data lives in ../palettes/ relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    brand_dir = os.path.join(script_dir, '..', 'palettes', 'flagship')
    
    # Single theme mode
    if len(sys.argv) == 3 and sys.argv[1] == '--single':
        slug = sys.argv[2]
        brand_path = os.path.join(brand_dir, f'{slug}.brand.json')
        
        with open(brand_path) as f:
            brand = json.load(f)
        
        # Construct a minimal theme_data dict from the brand JSON
        theme_data = {
            'perceptual_volume': brand.get('perceptual_volume', 0),
            'name': brand.get('name', ''),
            'provenance': {
                'source_title': brand.get('source', ''),
                'source_type': brand.get('source_type', ''),
            },
        }
        
        print(generate_description(brand, theme_data))
        sys.exit(0)
    
    # Batch mode — generate all 29
    results = []
    for fname in sorted(os.listdir(brand_dir)):
        if not fname.endswith('.brand.json'):
            continue
        slug = fname.replace('.brand.json', '')
        
        brand_path = os.path.join(brand_dir, fname)
        
        with open(brand_path) as f:
            brand = json.load(f)
        
        # Construct minimal theme_data from brand JSON
        theme_data = {
            'perceptual_volume': brand.get('perceptual_volume', 0),
            'name': brand.get('name', ''),
            'provenance': {
                'source_title': brand.get('source', ''),
                'source_type': brand.get('source_type', ''),
            },
        }
        
        desc = generate_description(brand, theme_data)
        results.append({
            'slug': slug,
            'name': brand['name'],
            'year': brand.get('year'),
            'collection': brand.get('collection', ''),
            'description': desc,
            'pv': brand.get('perceptual_volume', 0),
            'is_dark': hex_to_hsl(brand['roles']['Background']['hex'])[2] < 50,
            'best_for': [],
            'highlight_hex': brand['roles']['Highlight']['hex'],
            'support_hex': brand['roles']['Support']['hex'],
            'background_hex': brand['roles']['Background']['hex'],
            'canvas_hex': brand['roles']['Canvas']['hex'],
        })
    
    json.dump(results, sys.stdout, indent=2, ensure_ascii=False)