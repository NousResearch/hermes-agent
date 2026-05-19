"""
telegram_gateway/bot.py - Evelyn AI Web3 Companion Bot
========================================================
Commands:
  /start          - Welcome message (Evelyn intro)
  /pending        - List pending approval entries
  /approve <id>   - Approve a pending entry
  /reject <id>    - Reject a pending entry
  /status <id>    - Check entry status
  /contract <addr> - Analyze NFT contract
  /wallet <addr>  - Wallet summary
  /floor <slug>   - OpenSea floor price
  /risk <addr>    - AI risk analysis
  /clear          - Clear AI conversation history

AI Chat:
  Any normal text -> Evelyn responds with deep_waifu personality
  Auto-detects 0x addresses and OpenSea links

Inline buttons:
  Approve / Reject / Dry Run preview

SAFETY:
- Only TELEGRAM_ALLOWED_USERS can interact
- Private keys are NEVER shown or logged
- Bot does NOT auto-execute transactions
- AI chat CANNOT trigger blockchain transactions

Usage:
    python -m custom_tools.telegram_gateway.bot
"""

import os
import sys
import logging
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from custom_tools.approval_queue import (
    list_queue,
    approve,
    reject,
    get_entry,
    count_pending,
)
from custom_tools.telegram_gateway.ai_chat import (
    get_ai_response_with_queue_context,
    clear_conversation,
)
from custom_tools.telegram_gateway.web3_skills import (
    analyze_contract,
    analyze_wallet,
    get_floor_price,
    analyze_risk,
    detect_address,
    detect_opensea_slug,
    detect_chain_from_text,
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
    return "🚫 Maaf sayang, kamu belum terdaftar di TELEGRAM_ALLOWED_USERS."


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
            InlineKeyboardButton("👁 Preview", callback_data=f"preview_{entry_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


# === Command Handlers ===

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - Evelyn intro."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    pending = count_pending()
    pending_text = f"\n\n⏳ Ada {pending} pending approval nih." if pending > 0 else ""

    msg = (
        f"hai sayang 💕\n\n"
        f"aku <b>Evelyn</b>, AI companion kamu buat Web3/NFT.\n"
        f"mau ngobrol, cek contract, atau approve mint — tinggal bilang aja ya.\n\n"
        f"<b>Commands:</b>\n"
        f"  /pending - Cek antrian approval\n"
        f"  /approve &lt;id&gt; - Approve entry\n"
        f"  /reject &lt;id&gt; - Reject entry\n"
        f"  /contract &lt;addr&gt; - Analyze contract\n"
        f"  /wallet &lt;addr&gt; - Cek wallet\n"
        f"  /floor &lt;slug&gt; - Floor price\n"
        f"  /risk &lt;addr&gt; - Risk analysis\n"
        f"  /clear - Reset chat history\n\n"
        f"atau ketik apa aja, aku bales kok 😊"
        f"{pending_text}"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_pending(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pending command."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    entries = list_queue(status="pending", limit=10)

    if not entries:
        await update.message.reply_text("✅ Queue kosong sayang, santai dulu~")
        return

    await update.message.reply_text(
        f"⏳ ada {len(entries)} pending nih beb 😈",
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
        await update.message.reply_text("kasih ID-nya dong sayang~ \nUsage: /approve <id>")
        return

    try:
        entry_id = int(context.args[0])
        approve(entry_id, approved_by=f"telegram:{user_id}")
        await update.message.reply_text(
            f"siapp cintaaa 😈\nentry #{entry_id} udah aku approve ya ✅",
            parse_mode="HTML",
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ gagal beb: {e}")
    except Exception as e:
        await update.message.reply_text(f"❌ error: {e}")


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
        reject(entry_id, reason=reason)
        await update.message.reply_text(
            f"oke sayang, entry #{entry_id} aku reject ❌\nreason: {reason}",
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ gagal: {e}")
    except Exception as e:
        await update.message.reply_text(f"❌ error: {e}")


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
        await update.message.reply_text(f"❌ {e}")


async def cmd_contract(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /contract <address> [chain] command."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    if not context.args:
        await update.message.reply_text("kasih address-nya dong~\nUsage: /contract <0x...> [chain]")
        return

    address = context.args[0]
    chain = context.args[1] if len(context.args) > 1 else "ethereum"

    await update.message.chat.send_action("typing")
    result = await analyze_contract(address, chain)
    await update.message.reply_text(result, parse_mode="HTML", disable_web_page_preview=True)


async def cmd_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /wallet <address> [chain] command."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    if not context.args:
        await update.message.reply_text("Usage: /wallet <0x...> [chain]")
        return

    address = context.args[0]
    chain = context.args[1] if len(context.args) > 1 else "ethereum"

    await update.message.chat.send_action("typing")
    result = await analyze_wallet(address, chain)
    await update.message.reply_text(result, parse_mode="HTML")


async def cmd_floor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /floor <collection_slug> command."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    if not context.args:
        await update.message.reply_text("Usage: /floor <collection-slug>\nContoh: /floor boredapeyachtclub")
        return

    slug = context.args[0].lower().strip()
    # Extract slug from opensea URL if provided
    detected = detect_opensea_slug(slug)
    if detected:
        slug = detected

    await update.message.chat.send_action("typing")
    result = await get_floor_price(slug)
    await update.message.reply_text(result, parse_mode="HTML", disable_web_page_preview=True)


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /risk <contract_address> [chain] command."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    if not context.args:
        await update.message.reply_text("Usage: /risk <0x...> [chain]")
        return

    address = context.args[0]
    chain = context.args[1] if len(context.args) > 1 else "ethereum"

    await update.message.chat.send_action("typing")
    result = await analyze_risk(address, user_id, chain)

    # Split if too long
    if len(result) <= 4096:
        await update.message.reply_text(result, parse_mode="HTML", disable_web_page_preview=True)
    else:
        for i in range(0, len(result), 4096):
            await update.message.reply_text(result[i:i + 4096], parse_mode="HTML", disable_web_page_preview=True)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command - reset AI memory."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    clear_conversation(user_id)
    await update.message.reply_text("🧹 memory cleared sayang~ fresh start buat kita 💕")


# === AI Chat Handler (catches all non-command text) ===

async def handle_ai_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle normal text messages with AI + auto-detection."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text(unauthorized_message())
        return

    message_text = update.message.text
    if not message_text:
        return

    # === Auto-detection ===

    # Detect 0x address -> offer contract/wallet analysis
    detected_addr = detect_address(message_text)
    if detected_addr and message_text.strip() == detected_addr:
        # Pure address sent, offer options
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔍 Contract", callback_data=f"contract_{detected_addr}"),
                InlineKeyboardButton("👛 Wallet", callback_data=f"wallet_{detected_addr}"),
            ],
            [
                InlineKeyboardButton("⚠️ Risk", callback_data=f"risk_{detected_addr}"),
            ],
        ])
        await update.message.reply_text(
            f"aku detect address nih sayang~\n<code>{detected_addr}</code>\n\nmau aku cek apa?",
            parse_mode="HTML",
            reply_markup=keyboard,
        )
        return

    # Detect OpenSea link -> auto floor check
    opensea_slug = detect_opensea_slug(message_text)
    if opensea_slug and "opensea.io" in message_text:
        await update.message.chat.send_action("typing")
        result = await get_floor_price(opensea_slug)
        await update.message.reply_text(result, parse_mode="HTML", disable_web_page_preview=True)
        return

    # === Normal AI chat ===
    await update.message.chat.send_action("typing")
    response = await get_ai_response_with_queue_context(user_id, message_text)

    # Send response (split if too long)
    if len(response) <= 4096:
        await update.message.reply_text(response)
    else:
        for i in range(0, len(response), 4096):
            await update.message.reply_text(response[i:i + 4096])


# === Callback Query Handler (Inline Buttons) ===

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button presses."""
    query = update.callback_query
    user_id = query.from_user.id

    if not is_authorized(user_id):
        await query.answer("Unauthorized", show_alert=True)
        return

    data = query.data
    parts = data.split("_", 1)

    if len(parts) != 2:
        await query.answer("Invalid action")
        return

    action, param = parts[0], parts[1]

    try:
        # Approval actions
        if action == "approve":
            entry_id = int(param)
            approve(entry_id, approved_by=f"telegram:{user_id}")
            await query.answer(f"✅ #{entry_id} approved!")
            await query.edit_message_text(
                f"✅ <b>APPROVED</b> - Entry #{entry_id}\napproved by sayang 😈",
                parse_mode="HTML",
            )

        elif action == "reject":
            entry_id = int(param)
            reject(entry_id, reason=f"Rejected via button by user {user_id}")
            await query.answer(f"❌ #{entry_id} rejected")
            await query.edit_message_text(
                f"❌ <b>REJECTED</b> - Entry #{entry_id}",
                parse_mode="HTML",
            )

        elif action == "preview":
            entry_id = int(param)
            entry = get_entry(entry_id)
            text = format_entry_preview(entry)
            text += "\n\n🔍 <b>Preview only. No tx sent.</b>"
            await query.answer("Preview loaded")
            keyboard = get_approval_keyboard(entry_id)
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=keyboard)

        # Web3 skill actions (from auto-detection)
        elif action == "contract":
            await query.answer("Analyzing contract...")
            result = await analyze_contract(param, "ethereum")
            await query.edit_message_text(result, parse_mode="HTML", disable_web_page_preview=True)

        elif action == "wallet":
            await query.answer("Checking wallet...")
            result = await analyze_wallet(param, "ethereum")
            await query.edit_message_text(result, parse_mode="HTML")

        elif action == "risk":
            await query.answer("Analyzing risk...")
            result = await analyze_risk(param, user_id, "ethereum")
            if len(result) <= 4096:
                await query.edit_message_text(result, parse_mode="HTML", disable_web_page_preview=True)
            else:
                await query.edit_message_text(result[:4096], parse_mode="HTML", disable_web_page_preview=True)

        else:
            await query.answer("Unknown action")

    except ValueError as e:
        await query.answer(f"Error: {e}", show_alert=True)
    except Exception as e:
        await query.answer(f"Error: {str(e)[:100]}", show_alert=True)


# === Main ===

def main():
    """Start Evelyn - AI Web3 Companion Bot."""
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        print("Set in .env: TELEGRAM_BOT_TOKEN=your-bot-token")
        sys.exit(1)

    if not ALLOWED_USERS:
        print("ERROR: TELEGRAM_ALLOWED_USERS not set")
        print("Set in .env: TELEGRAM_ALLOWED_USERS=123456789")
        sys.exit(1)

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    personality = os.getenv("AI_PERSONALITY_MODE", "deep_waifu")

    print(f"╔══════════════════════════════════════╗")
    print(f"║   Evelyn - AI Web3 Companion Bot     ║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║  Personality: {personality:<22} ║")
    print(f"║  AI Chat: {'ENABLED' if openrouter_key else 'DISABLED':<25} ║")
    print(f"║  Model: {os.getenv('OPENROUTER_MODEL', 'gpt-4o-mini'):<27} ║")
    print(f"║  DRY_RUN: {str(DRY_RUN):<25} ║")
    print(f"║  Users: {str(ALLOWED_USERS):<27} ║")
    print(f"╚══════════════════════════════════════╝")
    print()

    app = Application.builder().token(BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("pending", cmd_pending))
    app.add_handler(CommandHandler("approve", cmd_approve))
    app.add_handler(CommandHandler("reject", cmd_reject))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("contract", cmd_contract))
    app.add_handler(CommandHandler("wallet", cmd_wallet))
    app.add_handler(CommandHandler("floor", cmd_floor))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("clear", cmd_clear))

    # Inline button handler
    app.add_handler(CallbackQueryHandler(button_callback))

    # AI chat handler (catches all non-command text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ai_message))

    print("Evelyn is online 💕 Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
