import os, io, base64, urllib.request
from PIL import Image

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    desktop_assets = os.path.join(root, "apps", "desktop", "assets")
    appx_dir = os.path.join(desktop_assets, "appx")
    packaging_dir = os.path.join(root, "apps", "desktop", "packaging")
    os.makedirs(appx_dir, exist_ok=True)
    os.makedirs(packaging_dir, exist_ok=True)

    src_file = os.path.join(desktop_assets, "unitalk-logo-source.png")
    if not os.path.exists(src_file):
        print(f"Source file not found at {src_file}. Please place 504x505 logoUnitalk.png there.")
        return

    im = Image.open(src_file).convert("RGBA")
    master_size, inner_size = 1024, 860
    resized = im.resize((inner_size, inner_size), Image.Resampling.LANCZOS)
    master = Image.new("RGBA", (master_size, master_size), (0, 0, 0, 0))
    offset = ((master_size - inner_size) // 2, (master_size - inner_size) // 2)
    master.paste(resized, offset, resized)

    master.save(os.path.join(desktop_assets, "icon.png"), "PNG", optimize=True)
    master.save(os.path.join(desktop_assets, "icon-dark.png"), "PNG", optimize=True)
    master.save(os.path.join(desktop_assets, "icon-mac.png"), "PNG", optimize=True)

    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    master.save(os.path.join(desktop_assets, "icon.ico"), format="ICO", sizes=ico_sizes)
    master.save(os.path.join(desktop_assets, "icon-dark.ico"), format="ICO", sizes=ico_sizes)

    master.save(os.path.join(desktop_assets, "icon.icns"), format="ICNS")
    master.save(os.path.join(desktop_assets, "icon-dark.icns"), format="ICNS")
    master.save(os.path.join(packaging_dir, "dmg-volume.icns"), format="ICNS")

    appx_specs = {
        "Square150x150Logo.png": (150, 150),
        "Square150x150Logo.scale-125.png": (188, 188),
        "Square150x150Logo.scale-150.png": (225, 225),
        "Square150x150Logo.scale-200.png": (300, 300),
        "Square150x150Logo.scale-400.png": (600, 600),
        "Square44x44Logo.png": (44, 44),
        "Square44x44Logo.scale-125.png": (55, 55),
        "Square44x44Logo.scale-150.png": (66, 66),
        "Square44x44Logo.scale-200.png": (88, 88),
        "Square44x44Logo.scale-400.png": (176, 176),
        "StoreLogo.png": (50, 50),
        "StoreLogo.scale-125.png": (63, 63),
        "StoreLogo.scale-150.png": (75, 75),
        "StoreLogo.scale-200.png": (100, 100),
        "StoreLogo.scale-400.png": (200, 200),
    }
    for sz in [16, 20, 24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 96, 256]:
        appx_specs[f"Square44x44Logo.targetsize-{sz}.png"] = (sz, sz)
        appx_specs[f"Square44x44Logo.targetsize-{sz}_altform-unplated.png"] = (sz, sz)
        appx_specs[f"Square44x44Logo.targetsize-{sz}_altform-lightunplated.png"] = (sz, sz)

    for fname, (w, h) in appx_specs.items():
        master.resize((w, h), Image.Resampling.LANCZOS).save(os.path.join(appx_dir, fname), "PNG", optimize=True)

    for suffix, (ww, hh) in [("", (310, 150)), (".scale-125", (388, 188)), (".scale-150", (465, 225)), (".scale-200", (620, 300)), (".scale-400", (1240, 600))]:
        wide = Image.new("RGBA", (ww, hh), (0, 0, 0, 0))
        lw = int(hh * 0.8)
        lt = master.resize((lw, lw), Image.Resampling.LANCZOS)
        wide.paste(lt, ((ww - lw) // 2, (hh - lw) // 2), lt)
        wide.save(os.path.join(appx_dir, f"Wide310x150Logo{suffix}.png"), "PNG", optimize=True)

    print("All Unitalk desktop assets generated successfully.")

if __name__ == "__main__":
    main()
