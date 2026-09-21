"""Generate lightweight HTML previews for X/Twitter article bundles.

Resizes images to 800px width JPEG q80 before base64 embedding so each
preview stays under 8MB for Discord attachment.
"""
import base64
import io
import re
from pathlib import Path

BUNDLES_DIR = Path("/home/kensei/repos/KenseiAgent/content_engine/output/articles")
OUT_DIR = Path("/home/kensei/.hermes/reports/blog-previews")
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Install with: pip install Pillow --break-system-packages")
    raise


def embed_image_resized(path: Path, max_width: int = 800, quality: int = 80) -> str:
    """Resize image and return as base64 JPEG data URI."""
    if not path.exists():
        return ""
    img = Image.open(path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def markdown_to_html(md: str, bundle_dir: Path) -> str:
    """Convert article markdown to HTML, embedding resized images."""
    def replace_img(m):
        alt = m.group(1)
        img_path = bundle_dir / "imgs" / m.group(2)
        data_uri = embed_image_resized(img_path)
        if data_uri:
            return f'<img src="{data_uri}" alt="{alt}" style="width:100%;border-radius:8px;margin:16px 0;" />'
        return f'<div style="padding:20px;text-align:center;color:#888;border:1px dashed #444;margin:16px 0;">[IMAGE MISSING]</div>'

    md = re.sub(r'!\[([^\]]*)\]\(imgs/([^)]+)\)', replace_img, md)

    lines = md.split("\n")
    html_lines = []
    in_list = False
    for line in lines:
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line[2:]}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            line = re.sub(r'`([^`]+)`', r'<code>\1</code>', line)
            if line.strip():
                html_lines.append(f"<p>{line}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def build_preview(bundle_dir: Path) -> str:
    md = (bundle_dir / "article.md").read_text()
    title = md.split("\n")[0].replace("# ", "").strip()
    body_html = markdown_to_html(md, bundle_dir)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>X Article Preview: {title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: #0d1117; color: #e6edf3;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  line-height: 1.7; max-width: 800px; margin: 0 auto; padding: 20px;
}}
h1 {{ font-size: 1.8em; margin: 20px 0; color: #58a6ff; }}
h2 {{ font-size: 1.3em; margin: 24px 0 12px; color: #79c0ff; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
h3 {{ font-size: 1.1em; margin: 20px 0 8px; color: #d2a8ff; }}
p {{ margin: 12px 0; }}
code {{ background: #161b22; padding: 2px 6px; border-radius: 4px; font-family: monospace; color: #f0883e; }}
ul {{ margin: 12px 0; padding-left: 24px; }}
li {{ margin: 4px 0; }}
strong {{ color: #f0f6fc; }}
img {{ width: 100%; border-radius: 8px; margin: 16px 0; border: 1px solid #21262d; }}
.preview-banner {{
  background: linear-gradient(90deg, #238636, #1f6feb);
  padding: 8px 16px; text-align: center; font-size: 12px; font-weight: 600;
  color: #fff; border-radius: 6px; margin-bottom: 20px;
}}
</style>
</head>
<body>
<div class="preview-banner">X/TWITTER ARTICLE PREVIEW - not published</div>
{body_html}
</body>
</html>"""


def main():
    bundles = sorted(BUNDLES_DIR.glob("2*"))
    bundles = [b for b in bundles if (b / "article.md").exists()]
    print(f"Generating lightweight previews for {len(bundles)} article bundles")
    for b in bundles:
        slug = b.name
        out_path = OUT_DIR / f"xarticle-{slug}.html"
        html = build_preview(b)
        out_path.write_text(html, encoding="utf-8")
        size_kb = out_path.stat().st_size // 1024
        size_mb = size_kb // 1024
        ok = "✓" if size_kb < 8 * 1024 else "✗ OVERSIZE"
        print(f"  {ok} {slug} -> {size_kb}KB ({size_mb}MB)")
    print(f"\nDone. {len(bundles)} previews in {OUT_DIR}")


if __name__ == "__main__":
    main()