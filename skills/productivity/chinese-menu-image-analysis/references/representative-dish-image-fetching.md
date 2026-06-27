# Representative Dish Image Fetching

Use this reference when enriching a Chinese menu HTML report with per-dish representative images.

## Goal

Add useful visual context without implying the image is from the restaurant. Prefer safe, attributable sources and keep the report usable even when image search is incomplete.

## Source and labeling rules

- Label every image as one of: `menu-crop`, `official`, `representative`, `generated`, or `placeholder`.
- Use `representative` for public images that show the dish/category but are not from the target restaurant.
- Keep `source`, `credit`, `license` when available.
- If no reliable image is found quickly, use a placeholder card. Do not block the whole menu report.
- Include the Korean disclaimer:

```text
음식 사진은 실제 매장 제공 이미지가 아니라 메뉴명 기반의 대표/참고 이미지입니다. 실제 제공 모양과 다를 수 있습니다.
```

## Wikimedia Commons API pattern

Wikimedia Commons may return HTTP 403 for default Python/urlopen user agents. Always send a descriptive `User-Agent`.

```python
import json, urllib.parse, urllib.request

UA = {
    "User-Agent": "HermesAgentMenuAnalysis/1.0 (local report generation; representative dish images)"
}

query = "xiao long bao"
api = (
    "https://commons.wikimedia.org/w/api.php?action=query"
    "&generator=search"
    f"&gsrsearch={urllib.parse.quote(query)}"
    "&gsrnamespace=6"
    "&gsrlimit=3"
    "&prop=imageinfo"
    "&iiprop=url|extmetadata"
    "&iiurlwidth=640"
    "&format=json"
    "&origin=*"
)
req = urllib.request.Request(api, headers=UA)
data = json.load(urllib.request.urlopen(req, timeout=12))
```

For each result:

- Prefer `imageinfo[0].thumburl` over full-size `url` for compact reports.
- Skip `.svg` and `.gif` when the HTML report expects photo-like thumbnails.
- Use `imageinfo[0].descriptionurl` as the source link.
- Pull `Artist` and `LicenseShortName` from `extmetadata` if present; strip HTML tags from `Artist`.

## Query fallback strategy

For each OCR-confirmed dish:

1. Exact Chinese menu name.
2. Simplified/Traditional variant if obvious.
3. Common English/Korean name or pinyin.
4. Regional/category fallback.

Examples:

- `鮮肉小籠包` → `xiao long bao`
- `酒湯生煎包` → `shengjian mantou`
- `外婆紅燒肉` → `hong shao rou braised pork belly`
- `乾煸四季豆` → `dry fried green beans Sichuan`

If all queries fail, keep a placeholder with the attempted query so the user understands what was tried.

## Single-file HTML reports

For Discord/mobile delivery, inline local images as `data:` URLs when practical. This avoids broken relative image paths when the file is downloaded or attached separately.

Verification snippet:

```python
from html.parser import HTMLParser
from pathlib import Path

class ImgParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.imgs = []
    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.imgs.append(dict(attrs).get("src", ""))

p = Path("menu_analysis_with_images.html")
parser = ImgParser()
parser.feed(p.read_text(encoding="utf-8"))
non_data = [src for src in parser.imgs if not src.startswith("data:image/")]
print("img_tags", len(parser.imgs))
print("non_data_src_count", len(non_data))
```

For a self-contained attachment, `non_data_src_count` should be `0`. For a served report with relative assets, verify every relative URL returns HTTP 200 from the served location.
