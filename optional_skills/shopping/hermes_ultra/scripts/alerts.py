"""Rich terminal alerts and notification system.

Provides beautifully formatted terminal output for price alerts,
deal scores, scalper warnings, and trend reports. Uses the ``rich``
library (already a hermes-agent dependency).

Notification hooks include OS-aware Windows toast support:
Windows hook is only enabled on Windows systems — on other platforms
it silently falls back to terminal-only output.
"""

import logging
import platform
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notification hook system (modular, extensible)
# ---------------------------------------------------------------------------

class NotificationHook(ABC):
    """Base class for notification delivery hooks."""

    @abstractmethod
    def send(self, title: str, message: str, level: str = "info") -> bool:
        """Deliver a notification.

        Args:
            title: Alert title.
            message: Full rendered message (may contain ANSI or markdown).
            level: One of 'info', 'warning', 'critical'.

        Returns:
            True if delivery succeeded.
        """


class TerminalHook(NotificationHook):
    """Default hook — prints to stdout via Rich."""

    def send(self, title: str, message: str, level: str = "info") -> bool:
        try:
            from rich.console import Console
            console = Console()
            console.print(message)
            return True
        except ImportError:
            print(message)
            return True


class WindowsNotificationHook(NotificationHook):
    """Sends native Windows toast notifications via PowerShell.

    Only active on Windows — gracefully returns True on other platforms.
    """

    def __init__(self) -> None:
        self._is_windows: bool = platform.system() == "Windows"

    def send(self, title: str, message: str, level: str = "info") -> bool:
        if not self._is_windows:
            return True  # Silent no-op on non-Windows

        if level == "info":
            return True  # Don't spam desktop for info pings

        import subprocess

        # Clean ANSI codes for system toast
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-2]*[amkqtyurHW])")
        clean_msg = ansi_escape.sub("", message).strip()
        clean_msg = " / ".join(clean_msg.splitlines()[:3]).replace('"', "'")

        ps_script = f"""
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
        $textNodes = $template.GetElementsByTagName("text")
        $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) | Out-Null
        $textNodes.Item(1).AppendChild($template.CreateTextNode("{clean_msg}")) | Out-Null
        $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("HermesUltra").Show($toast)
        """
        try:
            subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True, check=False,
            )
            return True
        except Exception:
            return False


# Global hook list — extensions append to this
_hooks: List[NotificationHook] = [TerminalHook(), WindowsNotificationHook()]


def add_notification_hook(hook: NotificationHook) -> None:
    """Register an additional notification hook."""
    _hooks.append(hook)


def _notify_all(title: str, message: str, level: str = "info") -> None:
    """Send a notification through all registered hooks."""
    for hook in _hooks:
        try:
            hook.send(title, message, level)
        except Exception as e:
            logger.warning("Notification hook %s failed: %s", type(hook).__name__, e)


# ---------------------------------------------------------------------------
# Alert formatters (return Rich-renderable strings)
# ---------------------------------------------------------------------------

def format_deal_alert(
    product_name: str,
    current_price: float,
    target_price: Optional[float],
    score: int,
    score_label: str,
    url: str = "",
) -> str:
    """Format a deal score alert as a Rich panel."""
    try:
        from rich.panel import Panel
        from rich.console import Console
        from io import StringIO

        lines = []
        lines.append(f"[bold cyan]📦 {product_name}[/bold cyan]")
        lines.append("")
        lines.append(f"💰 Price: [bold green]${current_price:,.2f}[/bold green]")
        if target_price:
            if current_price <= target_price:
                diff = target_price - current_price
                lines.append(f"🎯 Target: ${target_price:,.2f} [bold green](✅ ${diff:,.2f} below!)[/bold green]")
            else:
                diff = current_price - target_price
                lines.append(f"🎯 Target: ${target_price:,.2f} [yellow](${diff:,.2f} above)[/yellow]")

        filled = score // 5
        empty = 20 - filled
        if score >= 80:
            bar_color = "green"
        elif score >= 60:
            bar_color = "cyan"
        elif score >= 30:
            bar_color = "yellow"
        else:
            bar_color = "red"
        bar = f"[{bar_color}]{'█' * filled}{'░' * empty}[/{bar_color}] {score}/100"
        lines.append(f"\n📊 Deal Score: {bar}")
        lines.append(f"   {score_label}")

        if url:
            lines.append(f"\n🔗 {url}")

        content = "\n".join(lines)

        if score >= 80:
            border, title = "bold green", "🔥 UNMISSABLE DEAL!"
        elif score >= 60:
            border, title = "bold cyan", "👍 GOOD DEAL"
        elif score >= 30:
            border, title = "yellow", "🤔 FAIR DEAL"
        else:
            border, title = "red", "💀 BAD DEAL"

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=70)
        panel = Panel(content, title=title, border_style=border, width=70, padding=(1, 2))
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        return _plain_deal_alert(product_name, current_price, target_price, score, score_label, url)


def _plain_deal_alert(
    name: str, price: float, target: Optional[float],
    score: int, label: str, url: str,
) -> str:
    """Plain-text fallback for deal alerts."""
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        f"║  🔔 DEAL ALERT — Score: {score}/100",
        "╠══════════════════════════════════════════════════════╣",
        f"║  📦 {name}",
        f"║  💰 Price: ${price:,.2f}",
    ]
    if target:
        lines.append(f"║  🎯 Target: ${target:,.2f}")
    lines.append(f"║  📊 {label}")
    if url:
        lines.append(f"║  🔗 {url}")
    lines.append("╚══════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def format_scalper_warning(
    product_name: str,
    risk_level: str,
    risk_emoji: str,
    deviation_pct: float,
    avg_price: float,
    current_price: float,
    analysis_text: str,
) -> str:
    """Format a scalper/overpricing warning."""
    try:
        from rich.panel import Panel
        from rich.console import Console
        from io import StringIO

        lines = [
            "[bold red]⚠️ SCALPER ANALYSIS[/bold red]",
            f"\n📦 {product_name}",
            f"💰 Current: [bold]${current_price:,.2f}[/bold]",
            f"📊 Average: ${avg_price:,.2f}",
            f"📈 Deviation: [bold red]+{deviation_pct:.1f}%[/bold red]",
            f"\n{risk_emoji} Risk Level: [bold]{risk_level}[/bold]",
            f"\n{analysis_text}",
        ]

        content = "\n".join(lines)

        if risk_level == "HIGH":
            border, title = "bold red", "🚨 SCALPER WARNING!"
        elif risk_level == "MEDIUM":
            border, title = "bold yellow", "⚠️ SUSPICIOUS PRICE"
        else:
            border, title = "green", "✅ NORMAL PRICE"

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=70)
        panel = Panel(content, title=title, border_style=border, width=70, padding=(1, 2))
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        return f"[{risk_level}] {product_name}: {analysis_text}"


def format_price_table(products: list) -> str:
    """Format a list of tracked products as a Rich table."""
    try:
        from rich.table import Table
        from rich.console import Console
        from io import StringIO

        table = Table(
            title="📋 Tracked Products",
            show_header=True, header_style="bold cyan", border_style="dim",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Product", style="bold", max_width=30)
        table.add_column("Price", justify="right", style="green")
        table.add_column("Target", justify="right")
        table.add_column("Stock", justify="center")
        table.add_column("Site", style="dim")

        stock_icons = {
            "in_stock": "✅", "out_of_stock": "❌",
            "limited": "⚠️", "unknown": "❓",
        }

        for i, p in enumerate(products, 1):
            price_str = f"${p.get('current_price', 0):,.2f}" if p.get("current_price") else "—"
            target_str = f"${p.get('target_price', 0):,.2f}" if p.get("target_price") else "—"
            stock = stock_icons.get(p.get("stock_status", "unknown"), "❓")
            name = (p.get("name", "Unknown"))[:30]
            table.add_row(str(i), name, price_str, target_str, stock, p.get("site", ""))

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=90)
        console.print(table)
        return buf.getvalue()

    except ImportError:
        lines = ["Tracked Products:"]
        for i, p in enumerate(products, 1):
            lines.append(f"  {i}. {p.get('name', '?')} — ${p.get('current_price', '?')}")
        return "\n".join(lines)


def format_market_overview(results: list, product_name: str = "") -> str:
    """Format a multi-source price comparison as a Rich table."""
    if not results:
        return ""

    priced = [r for r in results if r.get("price") and r["price"] > 0]
    if not priced:
        return "🔍 Scanning global markets... All sources currently shielded. Retrying in background."

    best_price = min(priced, key=lambda r: r["price"])["price"]

    try:
        from rich.table import Table
        from rich.console import Console
        from io import StringIO

        table = Table(show_header=True, header_style="bold cyan", border_style="dim", pad_edge=True)
        table.add_column("Store", style="bold", min_width=12)
        table.add_column("Price", justify="right", min_width=12)
        table.add_column("Stock", justify="center", min_width=8)
        table.add_column("Note", min_width=16)

        stock_icons = {
            "in_stock": "✅", "out_of_stock": "❌ OOS",
            "limited": "⚠️", "unknown": "❓",
        }

        for r in priced:
            site = r.get("site", "?")
            price = r["price"]
            stock = r.get("stock_status", "unknown")
            price_str = f"[bold green]${price:,.2f}[/bold green]"
            stock_str = stock_icons.get(stock, "❓")
            note = "[bold green]✅ BEST PRICE[/bold green]" if price == best_price else ""
            table.add_row(site, price_str, stock_str, note)

        lines = []
        if product_name:
            lines.append(f"[bold cyan]📦 {product_name}[/bold cyan]\n")

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=72)
        if lines:
            console.print("\n".join(lines))
        console.print(table)

        if len(priced) >= 2:
            best_site = next(r["site"] for r in priced if r["price"] == best_price)
            max_price = max(r["price"] for r in priced)
            spread = max_price - best_price
            spread_pct = (spread / best_price) * 100 if best_price > 0 else 0
            console.print(f"\n💰 Best Price: [bold green]${best_price:,.2f}[/bold green] ({best_site})")
            console.print(f"📊 Price Spread: ${spread:,.2f} ({spread_pct:.1f}%)")
        else:
            best_site = next(r["site"] for r in priced if r["price"] == best_price)
            console.print(f"\n💰 Best Price: [bold green]${best_price:,.2f}[/bold green] ({best_site})")

        return buf.getvalue()

    except ImportError:
        lines = [f"  🏪 MARKET OVERVIEW — {product_name}", ""]
        for r in priced:
            lines.append(f"  {r.get('site', '?'):12s}  ${r['price']:>10,.2f}  {r.get('stock_status', '?')}")
        return "\n".join(lines)


def format_trend_report(
    product_name: str,
    direction: str,
    direction_emoji: str,
    predicted_change_pct: float,
    predicted_price: Optional[float],
    confidence: str,
    analysis_text: str,
    volatility_warning: bool = False,
) -> str:
    """Format a trend prediction report with volatility support."""
    try:
        from rich.panel import Panel
        from rich.console import Console
        from io import StringIO

        lines = [
            f"[bold]📊 Trend Analysis: {product_name}[/bold]",
            "",
            f"{direction_emoji} Direction: [bold]{direction}[/bold]",
            f"📈 Projected change: {predicted_change_pct:+.1f}%",
        ]
        if volatility_warning:
            lines.insert(2, "[bold red]⚠️ HIGH VOLATILITY - USE CAUTION[/bold red]")
        if predicted_price:
            lines.append(f"💰 Projected price: ${predicted_price:,.2f}")
        lines.append(f"🎯 Confidence: {confidence}")
        lines.append(f"\n{analysis_text}")

        content = "\n".join(lines)
        border = "red" if volatility_warning else ("cyan" if direction == "DOWN" else "yellow" if direction == "UP" else "dim")
        title = f"{direction_emoji} TREND PREDICTION"

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=70)
        panel = Panel(content, title=title, border_style=border, width=70, padding=(1, 2))
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        warn = " [HIGH VOLATILITY]" if volatility_warning else ""
        return f"[{direction}]{warn} {product_name}: {analysis_text}"


def format_full_report(
    product_name: str,
    current_price: float,
    target_price: Optional[float],
    deal_score: int,
    deal_label: str,
    scalper_risk: str,
    scalper_text: str,
    trend_direction: str,
    trend_text: str,
    url: str = "",
    market_prices: Optional[list] = None,
    volatility_warning: bool = False,
    reasoning_text: str = "",
) -> str:
    """Format a comprehensive intelligence report.

    Includes: Market Comparison, Savings Highlight, BUY NOW / WAIT
    recommendation, Scalper & Trend intelligence, and Reasoning.
    """
    try:
        from rich.panel import Panel
        from rich.console import Console
        from io import StringIO

        header = """
   ██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗
   ██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝
   ███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗
   ██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║
   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║
   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝
              ⚡ U L T R A  A L A R M ⚡"""

        lines = [
            f"[bold gold1]{header}[/bold gold1]", "",
            f"[bold cyan]📦 {product_name}[/bold cyan]",
            f"💰 Best Price: [bold green]${current_price:,.2f}[/bold green]",
        ]
        if target_price:
            if current_price <= target_price:
                diff = target_price - current_price
                lines.append(f"🎯 Target: ${target_price:,.2f} [bold green](✅ ${diff:,.2f} below!)[/bold green]")
            else:
                diff = current_price - target_price
                lines.append(f"🎯 Target: ${target_price:,.2f} [yellow](${diff:,.2f} above)[/yellow]")

        if volatility_warning:
            lines.append("\n[bold red]⚠️ HIGH VOLATILITY - USE CAUTION[/bold red]")

        # Market comparison
        if market_prices:
            priced = [r for r in market_prices if r.get("price") and r["price"] > 0]
            lines.append(f"\n{'─' * 50}")
            if priced:
                lines.append("[bold]🏪 Market Comparison[/bold]")
                for r in priced:
                    site = r.get("site", "?")
                    price = r["price"]
                    badge = " [bold green]← BEST[/bold green]" if price == current_price else ""
                    lines.append(f"   {site:12s}  [green]${price:>10,.2f}[/green]{badge}")
                if len(priced) >= 2:
                    avg_price = sum(r["price"] for r in priced) / len(priced)
                    savings_vs_avg = avg_price - current_price
                    savings_pct = (savings_vs_avg / avg_price) * 100 if avg_price > 0 else 0
                    lines.append("")
                    if savings_vs_avg > 0:
                        lines.append(f"💎 [bold green]You save ${savings_vs_avg:,.2f} ({savings_pct:.1f}%) vs market average (${avg_price:,.2f})[/bold green]")
                    elif savings_vs_avg < 0:
                        lines.append(f"⚠️ [yellow]${abs(savings_vs_avg):,.2f} ({abs(savings_pct):.1f}%) above market average (${avg_price:,.2f})[/yellow]")

        # Deal score bar
        lines.append(f"\n{'─' * 50}")
        filled = deal_score // 5
        empty = 20 - filled
        bar_color = "green" if deal_score >= 80 else "cyan" if deal_score >= 60 else "yellow" if deal_score >= 30 else "red"
        bar = f"[{bar_color}]{'█' * filled}{'░' * empty}[/{bar_color}] {deal_score}/100"
        lines.append(f"📊 Deal Score: {bar}")
        lines.append(f"   {deal_label}")

        # Recommendation
        lines.append(f"\n{'─' * 50}")
        if deal_score >= 80:
            lines.extend(["[bold green]🟢 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]",
                           "[bold green]   ✅ RECOMMENDATION: BUY NOW                  [/bold green]",
                           "[bold green]🟢 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]"])
        elif deal_score >= 60:
            lines.extend(["[bold cyan]🔵 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]",
                           "[bold cyan]   👍 RECOMMENDATION: GOOD DEAL — Consider It   [/bold cyan]",
                           "[bold cyan]🔵 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]"])
        elif deal_score >= 30:
            lines.extend(["[yellow]🟡 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/yellow]",
                           "[yellow]   ⏳ RECOMMENDATION: WAIT                       [/yellow]",
                           "[yellow]🟡 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/yellow]"])
        else:
            lines.extend(["[bold red]🔴 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]",
                           "[bold red]   🚫 RECOMMENDATION: DO NOT BUY               [/bold red]",
                           "[bold red]🔴 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]"])

        # Scalper + Trend
        lines.append(f"\n{'─' * 50}")
        lines.append(f"🔍 Scalper Analysis: [bold]{scalper_risk}[/bold]")
        lines.append(f"   {scalper_text}")
        lines.append(f"\n{'─' * 50}")
        lines.append(f"📈 Trend: [bold]{trend_direction}[/bold]")
        lines.append(f"   {trend_text}")

        # Reasoning layer (new)
        if reasoning_text:
            lines.append(f"\n{'─' * 50}")
            lines.append("[bold]🧠 Price Reasoning[/bold]")
            lines.append(f"   {reasoning_text}")

        if url:
            lines.append(f"\n🔗 {url}")

        content = "\n".join(lines)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=72)
        panel = Panel(
            content,
            title="[bold gold1]⚡ HERMES ULTRA — INTELLIGENCE REPORT ⚡[/bold gold1]",
            border_style="bold gold1", width=72, padding=(1, 2),
        )
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        lines = [
            "═" * 55,
            "  ⚡ HERMES ULTRA — INTELLIGENCE REPORT ⚡",
            "═" * 55,
            f"  📦 {product_name}",
            f"  💰 Best Price: ${current_price:,.2f}",
            f"  📊 Score: {deal_score}/100 — {deal_label}",
        ]
        if deal_score >= 80:
            lines.append("  ✅ RECOMMENDATION: BUY NOW")
        else:
            lines.append("  ⏳ RECOMMENDATION: WAIT")
        lines.extend([
            f"  🔍 Scalper: {scalper_risk} — {scalper_text}",
            f"  📈 Trend: {trend_direction} — {trend_text}",
        ])
        if reasoning_text:
            lines.append(f"  🧠 Reasoning: {reasoning_text}")
        lines.append("═" * 55)
        return "\n".join(lines)


def format_cross_site_scalper_alert(
    product_name: str,
    min_price: float,
    min_site: str,
    max_price: float,
    max_site: str,
    spread_pct: float,
) -> str:
    """Format a cross-site price spread warning."""
    try:
        from rich.panel import Panel
        from rich.console import Console
        from io import StringIO

        lines = [
            "[bold yellow]⚠️ CROSS-SITE PRICE ANOMALY[/bold yellow]",
            f"\n📦 {product_name}",
            f"\n💚 Cheapest: [bold green]${min_price:,.2f}[/bold green] ({min_site})",
            f"💔 Most Expensive: [bold red]${max_price:,.2f}[/bold red] ({max_site})",
            f"\n📊 Price Spread: [bold red]{spread_pct:.1f}%[/bold red]",
            "\n⚠️ Large price differences across stores may indicate scalper pricing,",
            "   regional restrictions, or different product versions.",
        ]
        content = "\n".join(lines)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=70)
        panel = Panel(
            content, title="🔍 CROSS-SITE ANALYSIS",
            border_style="bold yellow", width=70, padding=(1, 2),
        )
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        return (
            f"⚠️ CROSS-SITE ANOMALY: {product_name}\n"
            f"  Cheapest: ${min_price:,.2f} ({min_site})\n"
            f"  Most Expensive: ${max_price:,.2f} ({max_site})\n"
            f"  Spread: {spread_pct:.1f}%"
        )
