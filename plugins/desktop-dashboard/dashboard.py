"""desktop-dashboard: デスクトップ常駐ダッシュボード (時計/天気/トレンド)"""

from __future__ import annotations

import json
import math
import threading
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, Button, Canvas, Frame, Label, LEFT, RIGHT, StringVar, Tk, X

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ---------------------------------------------------------------------------
# 設定 (ユーザーが書き換え可能)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "city": "日野市",
    "latitude": 35.6693,
    "longitude": 139.3956,
    "weather_refresh_ms": 15 * 60 * 1000,
    "trend_refresh_ms": 30 * 60 * 1000,
}

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return DEFAULT_CONFIG


CONFIG = load_config()

CITY = CONFIG["city"]
LATITUDE = CONFIG["latitude"]
LONGITUDE = CONFIG["longitude"]
TIMEZONE = "Asia/Tokyo"
WEATHER_REFRESH_MS = CONFIG["weather_refresh_ms"]
TREND_REFRESH_MS = CONFIG["trend_refresh_ms"]

# 色設定
BG = "#0b1220"
PANEL = "#111c2e"
PANEL_2 = "#16243a"
FG = "#f8fafc"
MUTED = "#9fb0c5"
ACCENT = "#7dd3fc"
GREEN = "#86efac"
ORANGE = "#fdba74"

TRANSLATIONS = {
    "ja": {
        "weather": "天気",
        "xtrends": "X 日本のトレンド",
        "aitrends": "AI関連トレンド",
        "refresh": "更新",
        "updated": "最終更新",
        "system_time": "システム時刻",
        "timezone": "タイムゾーン",
        "loading": "取得中…",
        "unavailable": "取得できません（後で再試行）",
        "x_fallback": "X公式トレンドAPI未接続。代替ニュースを表示",
        "close_hint": "Esc / 右クリックで終了　・　言語切替",
        "language": "EN",
    },
    "en": {
        "weather": "Weather",
        "xtrends": "X Japan Trends",
        "aitrends": "AI Trends",
        "refresh": "Refresh",
        "updated": "Updated",
        "system_time": "System time",
        "timezone": "Timezone",
        "loading": "Loading…",
        "unavailable": "Unavailable (retry later)",
        "x_fallback": "X official Trends API not connected; showing fallback news",
        "close_hint": "Esc / right-click to close · Hino weather · language switch",
        "language": "日本語",
    },
}


# ---------------------------------------------------------------------------
# HTTP ユーティリティ
# ---------------------------------------------------------------------------
def http_json(url: str, timeout: int = 12) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "HermesAgentDesktopDashboard/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_text(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "HermesAgentDesktopDashboard/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# データ整形
# ---------------------------------------------------------------------------
def weather_text(data: dict, lang: str) -> str:
    current = data.get("current", {})
    temp = current.get("temperature_2m")
    apparent = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    code = current.get("weather_code")

    ja_codes = {
        0: "快晴", 1: "晴れ", 2: "一部くもり", 3: "くもり", 45: "霧", 48: "霧",
        51: "弱い霧雨", 61: "雨", 63: "雨", 65: "強い雨", 71: "雪", 73: "雪",
        75: "大雪", 80: "にわか雨", 95: "雷雨"
    }
    en_codes = {
        0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Fog", 51: "Drizzle", 61: "Rain", 63: "Rain",
        65: "Heavy rain", 71: "Snow", 73: "Snow", 75: "Heavy snow",
        80: "Showers", 95: "Thunderstorm"
    }

    desc = (ja_codes if lang == "ja" else en_codes).get(code, "—")
    if temp is None:
        return "—"
    if lang == "ja":
        return f"{desc}\n気温 {temp:.1f}°C / 体感 {apparent:.1f}°C\n湿度 {humidity}%"
    return f"{desc}\nTemp {temp:.1f}°C / feels {apparent:.1f}°C\nHumidity {humidity}%"


def extract_trends24(html: str, limit: int = 8):
    import re
    items = []
    for item in re.findall(r"(?:trend-name|trend-link)[^>]*>([^<]{1,80})<", html, flags=re.I):
        item = " ".join(item.split())
        if item and item not in items and len(item) > 1:
            items.append(item)
    return items[:limit]


# ---------------------------------------------------------------------------
# ツールスキーマ
# ---------------------------------------------------------------------------
DASHBOARD_SCHEMA = {
    "name": "desktop_dashboard",
    "description": "デスクトップ常駐ダッシュボード (デジタル/アナログ時計、天気、X日本トレンド、AI関連トレンド、i18n、OS時刻同期) の起動・停止・状態確認",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "status"],
                "description": "実行アクション: start=起動, stop=停止, status=状態確認"
            }
        },
        "required": ["action"],
    },
}


def check_available() -> bool:
    """tkinter が利用可能かどうか"""
    try:
        import tkinter
        tkinter.Tk()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ダッシュボード本体
# ---------------------------------------------------------------------------
class Dashboard:
    def __init__(self):
        self.root = Tk()
        self.root.title("Hermes Agent Dashboard")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.95)
        self.root.minsize(760, 520)
        self.root.geometry("900x650+40+40")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", self.close)

        self.lang = "ja"
        self.tz = datetime.now().astimezone().tzinfo
        self.local_tz_name = self.tz.tzname(datetime.now()) if self.tz else TIMEZONE

        self.weather_var = StringVar(value=self.t("loading"))
        self.x_var = StringVar(value=self.t("loading"))
        self.ai_var = StringVar(value=self.t("loading"))
        self.time_var = StringVar()
        self.date_var = StringVar()
        self.status_var = StringVar()
        self._refresh_timer_id = None

        self.build_ui()
        self.update_clock()
        self.refresh_all()
        # 確実に可視化・最前面化
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(50, lambda: (self.root.lift(), self.root.focus_force()))

    def t(self, key: str) -> str:
        return TRANSLATIONS[self.lang].get(key, key)

    def panel(self, parent, title: str, variable: StringVar, color=FG) -> Frame:
        box = Frame(parent, bg=PANEL_2, padx=16, pady=12, highlightthickness=1, highlightbackground="#243955")
        box.pack(side=LEFT, fill=BOTH, expand=True, padx=6, pady=6)
        Label(box, text=title, fg=color, bg=PANEL_2, font=("Segoe UI", 14, "bold"), anchor="w").pack(fill=X)
        Label(box, textvariable=variable, fg=FG, bg=PANEL_2, font=("Segoe UI", 12), justify=LEFT, anchor="nw", wraplength=330).pack(fill=BOTH, expand=True, pady=(10, 0))
        return box

    def build_ui(self):
        # ヘッダー
        header = Frame(self.root, bg=BG, padx=14, pady=10)
        header.pack(fill=X)
        Label(header, text="HERMES AGENT", fg=ACCENT, bg=BG, font=("Segoe UI", 12, "bold")).pack(side=LEFT)
        Button(header, text=self.t("language"), command=self.toggle_language, bg="#1f3553", fg=FG, relief="flat", padx=10).pack(side="right", padx=4)
        Button(header, text=self.t("refresh"), command=self.refresh_all, bg="#1f3553", fg=FG, relief="flat", padx=10).pack(side="right", padx=4)

        # 時計行
        clocks = Frame(self.root, bg=BG, padx=14)
        clocks.pack(fill=X)
        analog_box = Frame(clocks, bg=PANEL, padx=12, pady=12, highlightthickness=1, highlightbackground="#243955")
        analog_box.pack(side=LEFT, fill=BOTH, expand=True, padx=6, pady=6)
        Label(analog_box, text="ANALOG", fg=GREEN, bg=PANEL, font=("Segoe UI", 12, "bold")).pack()
        self.canvas = Canvas(analog_box, width=230, height=230, bg=PANEL, highlightthickness=0)
        self.canvas.pack()

        digital_box = Frame(clocks, bg=PANEL, padx=18, pady=24, highlightthickness=1, highlightbackground="#243955")
        digital_box.pack(side=LEFT, fill=BOTH, expand=True, padx=6, pady=6)
        Label(digital_box, textvariable=self.time_var, fg=FG, bg=PANEL, font=("Segoe UI", 38, "bold")).pack(pady=(28, 0))
        Label(digital_box, textvariable=self.date_var, fg=ACCENT, bg=PANEL, font=("Segoe UI", 13)).pack(pady=8)
        Label(digital_box, textvariable=self.status_var, fg=MUTED, bg=PANEL, font=("Segoe UI", 10), justify=LEFT).pack(pady=14)

        # データ行
        row1 = Frame(self.root, bg=BG, padx=14)
        row1.pack(fill=BOTH, expand=True)
        self.weather_box = self.panel(row1, self.t("weather") + f"  ·  {CITY}", self.weather_var, GREEN)
        self.x_box = self.panel(row1, self.t("xtrends"), self.x_var, ORANGE)

        row2 = Frame(self.root, bg=BG, padx=14)
        row2.pack(fill=BOTH, expand=True)
        self.ai_box = self.panel(row2, self.t("aitrends"), self.ai_var, ACCENT)
        self.info_box = self.panel(row2, "i18n / Sync", StringVar(value=""), MUTED)
        self.info_label = self.info_box.winfo_children()[-1]
        self.info_label.configure(text=f"{self.t('system_time')}: {self.local_tz_name}\n{self.t('close_hint')}")

        self.root.bind("<Button-3>", lambda e: self.close())

    def update_clock(self):
        now = datetime.now(self.tz)
        if self.lang == "ja":
            self.time_var.set(now.strftime("%H:%M:%S"))
            self.date_var.set(now.strftime("%Y年%m月%d日  %A"))
        else:
            self.time_var.set(now.strftime("%H:%M:%S"))
            self.date_var.set(now.strftime("%a, %b %d, %Y"))
        self.status_var.set(f"{self.t('timezone')}: {self.local_tz_name}\n{self.t('updated')}: {now.strftime('%H:%M:%S')}")
        self.draw_analog(now)
        self.root.after(250, self.update_clock)

    def draw_analog(self, now: datetime):
        c = self.canvas
        c.delete("all")
        cx, cy, r = 115, 115, 94
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=ACCENT, width=3)
        for i in range(12):
            angle = math.radians(i * 30 - 90)
            x1, y1 = cx + math.cos(angle) * (r - 8), cy + math.sin(angle) * (r - 8)
            x2, y2 = cx + math.cos(angle) * (r - 17), cy + math.sin(angle) * (r - 17)
            c.create_line(x1, y1, x2, y2, fill=FG, width=3)
        hands = [
            (((now.hour % 12) + now.minute / 60) * 30, 0.52, ORANGE, 5),
            ((now.minute + now.second / 60) * 6, 0.72, FG, 4),
            (now.second * 6, 0.82, GREEN, 2),
        ]
        for degrees, length, color, width in hands:
            angle = math.radians(degrees - 90)
            c.create_line(cx, cy, cx + math.cos(angle) * r * length, cy + math.sin(angle) * r * length, fill=color, width=width)
        c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=ACCENT, outline="")

    def refresh_all(self):
        if self._refresh_timer_id:
            self.root.after_cancel(self._refresh_timer_id)
        self.weather_var.set(self.t("loading"))
        self.x_var.set(self.t("loading"))
        self.ai_var.set(self.t("loading"))
        threading.Thread(target=self.fetch_weather, daemon=True).start()
        threading.Thread(target=self.fetch_x_trends, daemon=True).start()
        threading.Thread(target=self.fetch_ai_trends, daemon=True).start()
        self._refresh_timer_id = self.root.after(WEATHER_REFRESH_MS, self.refresh_all)

    def set_var(self, var: StringVar, value: str):
        self.root.after(0, lambda: var.set(value))

    def fetch_weather(self):
        try:
            params = urllib.parse.urlencode({
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code",
                "timezone": TIMEZONE
            })
            data = http_json(f"https://api.open-meteo.com/v1/forecast?{params}")
            self.set_var(self.weather_var, weather_text(data, self.lang))
        except Exception:
            self.set_var(self.weather_var, self.t("unavailable"))

    def fetch_x_trends(self):
        try:
            html = http_text("https://trends24.in/japan/")
            items = extract_trends24(html)
            if items:
                value = "\n".join(f"{i+1}. {x}" for i, x in enumerate(items))
                self.set_var(self.x_var, value)
            else:
                self.set_var(self.x_var, self.t("x_fallback"))
        except Exception:
            self.set_var(self.x_var, self.t("x_fallback"))

    def fetch_ai_trends(self):
        try:
            papers = http_json("https://huggingface.co/api/daily_papers?limit=8")
            titles = []
            for item in papers:
                paper = item.get("paper", item)
                title = paper.get("title")
                if title:
                    titles.append(title.replace("\n", " "))
            if titles:
                value = "\n".join(f"{i+1}. {x}" for i, x in enumerate(titles[:8]))
                self.set_var(self.ai_var, value)
            else:
                self.set_var(self.ai_var, self.t("unavailable"))
        except Exception:
            self.set_var(self.ai_var, self.t("unavailable"))

    def toggle_language(self):
        self.lang = "en" if self.lang == "ja" else "ja"
        self.weather_box.winfo_children()[0].configure(text=self.t("weather") + f"  ·  {CITY}")
        self.x_box.winfo_children()[0].configure(text=self.t("xtrends"))
        self.ai_box.winfo_children()[0].configure(text=self.t("aitrends"))
        self.info_label.configure(text=f"{self.t('system_time')}: {self.local_tz_name}\n{self.t('close_hint')}")
        self.refresh_all()

    def close(self, event=None):
        if self._refresh_timer_id:
            self.root.after_cancel(self._refresh_timer_id)
        self.root.destroy()


# ---------------------------------------------------------------------------
# グローバルインスタンス管理
# ---------------------------------------------------------------------------
_dashboard_instance: Dashboard | None = None
_dashboard_thread: threading.Thread | None = None


def _run_dashboard():
    global _dashboard_instance
    _dashboard_instance = Dashboard()
    _dashboard_instance.root.mainloop()


# ---------------------------------------------------------------------------
# ツールハンドラ
# ---------------------------------------------------------------------------
def handle_dashboard(action: str, task_id: str = None) -> str:
    global _dashboard_thread

    if action == "start":
        if _dashboard_thread and _dashboard_thread.is_alive():
            return json.dumps({"success": False, "message": "ダッシュボードは既に起動中です"})
        _dashboard_thread = threading.Thread(target=_run_dashboard, daemon=True)
        _dashboard_thread.start()
        return json.dumps({"success": True, "message": "デスクトップダッシュボードを起動しました (Esc / 右クリックで終了)"})

    elif action == "stop":
        if _dashboard_instance:
            _dashboard_instance.root.after(0, _dashboard_instance.close)
            _dashboard_instance = None
            return json.dumps({"success": True, "message": "ダッシュボードを停止しました"})
        return json.dumps({"success": False, "message": "ダッシュボードは起動していません"})

    elif action == "status":
        running = _dashboard_thread and _dashboard_thread.is_alive()
        return json.dumps({
            "success": True,
            "running": running,
            "message": "ダッシュボード起動中" if running else "ダッシュボードは停止しています"
        })

    return json.dumps({"success": False, "message": f"不明なアクション: {action}"})


# ---------------------------------------------------------------------------
# Slash コマンドハンドラ
# ---------------------------------------------------------------------------
def handle_slash_dashboard(args: str) -> str:
    action = args.strip().lower() or "status"
    import json
    result = handle_dashboard(action)
    data = json.loads(result)
    return data.get("message", str(data))


# ---------------------------------------------------------------------------
# CLI コマンド登録
# ---------------------------------------------------------------------------
def register_cli(parser):
    sub = parser.add_subparsers(dest="dashboard_action", required=True)
    sub.add_parser("start", help="ダッシュボードを起動")
    sub.add_parser("stop", help="ダッシュボードを停止")
    sub.add_parser("status", help="ダッシュボードの状態を表示")


def cli_main(args):
    import json
    result = handle_dashboard(args.dashboard_action)
    data = json.loads(result)
    print(data.get("message", str(data)))
    return 0


# ---------------------------------------------------------------------------
# 起動確認用
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _run_dashboard()