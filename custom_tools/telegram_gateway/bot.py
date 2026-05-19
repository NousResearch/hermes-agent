"""
telegram_gateway/bot.py - Telegram Approval Bot (Full Implementation)
======================================================================
Commands:
  /start       - Welcome message
  /pending     - List pending approval entries
  /approve <id> - Approve a pending entry
  /reject <id>  - Reject a pending entry
  /status <id>  - Check entry status

Inline buttons:
  Approve / Reject / Dry Run preview

SAFETY:
- Only TELEGRAM_ALLOWED_USERS can approve/reject
- Private keys are NEVER shown or logged
- Bot does NOT auto-execute transactions (approval only)
- DRY_RUN status shown in previews

Usage:
    python -m custom_tools.telegram_gateway.bot
"""

import os
import sys
import json
import logging
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from custom_tools.approval_queue import (
    list_queue,
    approve,
    reject,
    get_entry,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Configuration from environment
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USERS = [
    int(uid.strip())
    for uid in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")
    if uid.strip().isdigit()
]
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"


def is_authorized(user_id: int) -> bool:
    """Check if user is in allowed list."""
    return user_id in ALLOWED_USERS


def unauthorized_message() -> str:
    return "Unauthorized. Your user ID is not in TELEGRAM_ALLOWED_USERS."


def format_entry_preview(entry: dict) -> str:
    """Format a queue entry for Telegram display."""
    status_emoji = {
        "pending": "⏳",
        "approved": "✅",
        "rejected": "❌",
        "sent": "📤",
        "failed": "💥",
    }
    emoji = status_emoji.get(entry.get("status", ""), "❓")

    lines = [
        f"{emoji} <b>Entry #{entry['id']}</b> [{entry['status'].upper()}]",
        f"",
        f"<b>Chain:</b> {entry.get('chain', 'N/A')}",
        f"<b>Contract:</b> <code>{entry.get('contract_address', 'N/A')}</code>",
        f"<b>Wallet:</b> {entry.get('wallet_label', 'N/A')}",
        f"<b>Address:</b> <code>{entry.get('from_address', 'N/A')}</code>",
        f"<b>Function:</b> {entry.get('mint_function', 'N/A')}",
        f"<b>Quantity:</b> {entry.get('quantity', 'N/A')}",
        f"<b>Value:</b> {entry.get('total_value_wei', '0')} wei",
        f"<b>Gas Limit:</b> {entry.get('gas_limit', 'N/A')}",
        f"<b>Created:</b> {entry.get('created_at', 'N/A')}",
    ]

    if DRY_RUN:
        lines.append("")
        lines.append("⚠️ <b>DRY_RUN=true</b> - Execution will simulate only")

    return "\n".join(lines)


def get_approval_keyboard(entry_id: int) -> InlineKeyboardMarkup:
    """Get inline keyboard with Approve/Reject buttons."""
    keyboard = [
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"approve_{entry_id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject_{entry_id}"),
        ],
        [
            InlineKeyboardButton("👁 Dry Run Preview", callback_data=f"preview_{entry_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


# === Command Handlers ===

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    msg = (
        "🤖 <b>Hermes Web3 Approval Bot</b>\n\n"
        "Commands:\n"
        "  /pending - List pending approvals\n"
        "  /approve &lt;id&gt; - Approve entry\n"
        "  /reject &lt;id&gt; - Reject entry\n"
        "  /status &lt;id&gt; - Check entry status\n\n"
        f"DRY_RUN: <b>{'ON' if DRY_RUN else 'OFF'}</b>\n"
        f"Your User ID: <code>{user_id}</code>"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_pending(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pending command - list pending entries."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    entries = list_queue(status="pending", limit=10)

    if not entries:
        await update.message.reply_text("✅ No pending approvals.")
        return

    await update.message.reply_text(
        f"⏳ <b>{len(entries)} Pending Approval(s):</b>",
        parse_mode="HTML",
    )

    for entry in entries:
        text = format_entry_preview(entry)
        keyboard = get_approval_keyboard(entry["id"])
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)


async def cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /approve <id> command."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    if not context.args:
        await update.message.reply_text("Usage: /approve <id>")
        return

    try:
        entry_id = int(context.args[0])
        result = approve(entry_id, approved_by=f"telegram:{user_id}")
        await update.message.reply_text(
            f"✅ Entry #{entry_id} <b>APPROVED</b> by user {user_id}",
            parse_mode="HTML",
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ Error: {e}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_reject(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /reject <id> command."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    if not context.args:
        await update.message.reply_text("Usage: /reject <id> [reason]")
        return

    try:
        entry_id = int(context.args[0])
        reason = " ".join(context.args[1:]) if len(context.args) > 1 else f"Rejected by telegram:{user_id}"
        result = reject(entry_id, reason=reason)
        await update.message.reply_text(
            f"❌ Entry #{entry_id} <b>REJECTED</b>\nReason: {reason}",
            parse_mode="HTML",
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ Error: {e}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status <id> command."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    if not context.args:
        await update.message.reply_text("Usage: /status <id>")
        return

    try:
        entry_id = int(context.args[0])
        entry = get_entry(entry_id)
        text = format_entry_preview(entry)
        await update.message.reply_text(text, parse_mode="HTML")
    except ValueError as e:
        await update.message.reply_text(f"❌ Error: {e}")


# === Callback Query Handler (Inline Buttons) ===

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button presses."""
    query = update.callback_query
    user_id = query.from_user.id

    if not is_authorized(user_id):
        await query.answer("Unauthorized", show_alert=True)
        return

    data = query.data  # e.g. "approve_1", "reject_1", "preview_1"
    parts = data.split("_", 1)

    if len(parts) != 2:
        await query.answer("Invalid action")
        return

    action, entry_id_str = parts[0], parts[1]

    try:
        entry_id = int(entry_id_str)
    except ValueError:
        await query.answer("Invalid entry ID")
        return

    try:
        if action == "approve":
            approve(entry_id, approved_by=f"telegram:{user_id}")
            await query.answer(f"✅ Entry #{entry_id} APPROVED")
            await query.edit_message_text(
                f"✅ <b>APPROVED</b> - Entry #{entry_id}\nBy: user {user_id}",
                parse_mode="HTML",
            )

        elif action == "reject":
            reject(entry_id, reason=f"Rejected via Telegram button by user {user_id}")
            await query.answer(f"❌ Entry #{entry_id} REJECTED")
            await query.edit_message_text(
                f"❌ <b>REJECTED</b> - Entry #{entry_id}\nBy: user {user_id}",
                parse_mode="HTML",
            )

        elif action == "preview":
            entry = get_entry(entry_id)
            text = format_entry_preview(entry)
            text += "\n\n🔍 <b>This is a preview only. No transaction sent.</b>"
            await query.answer("Preview loaded")
            keyboard = get_approval_keyboard(entry_id)
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=keyboard)

        else:
            await query.answer("Unknown action")

    except ValueError as e:
        await query.answer(f"Error: {e}", show_alert=True)
    except Exception as e:
        await query.answer(f"Error: {e}", show_alert=True)


def main():
    """Start the Telegram bot."""
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in environment")
        print("Set it in .env: TELEGRAM_BOT_TOKEN=your-bot-token")
        sys.exit(1)

    if not ALLOWED_USERS:
        print("ERROR: TELEGRAM_ALLOWED_USERS not set in environment")
        print("Set it in .env: TELEGRAM_ALLOWED_USERS=123456789,987654321")
        sys.exit(1)

    print(f"Starting Hermes Telegram Approval Bot...")
    print(f"Allowed users: {ALLOWED_USERS}")
    print(f"DRY_RUN: {DRY_RUN}")
    print()

    app = Application.builder().token(BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("pending", cmd_pending))
    app.add_handler(CommandHandler("approve", cmd_approve))
    app.add_handler(CommandHandler("reject", cmd_reject))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CallbackQueryHandler(button_callback))

    # Start polling
    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
