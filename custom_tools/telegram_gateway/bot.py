"""
telegram_gateway/bot.py - Telegram Approval Bot (Placeholder)
==============================================================
Future implementation for Telegram-based transaction approval.

Functions:
- Receive approval requests
- Display tx preview in Telegram
- Approve/Reject inline buttons
- Only respond to TELEGRAM_ALLOWED_USERS

SAFETY:
- Only whitelisted user IDs can approve
- Bot token from environment variable
- Never exposes private keys or sensitive data

Usage (future):
    python -m custom_tools.telegram_gateway.bot
"""

import os
import sys
import json

# Telegram config from environment
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USERS = [
    int(uid.strip())
    for uid in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")
    if uid.strip().isdigit()
]



class TelegramApprovalBot:
    """
    Placeholder for Telegram approval bot.
    
    Future implementation will use python-telegram-bot or aiogram.
    
    Workflow:
    1. New transaction added to approval_queue
    2. Bot sends preview message to allowed users
    3. User taps Approve/Reject inline button
    4. Bot updates approval_queue status
    5. mint_executor picks up approved transactions
    """
    
    def __init__(self):
        self.token = BOT_TOKEN
        self.allowed_users = ALLOWED_USERS
        self.running = False
    
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is in allowed list."""
        return user_id in self.allowed_users
    
    def format_approval_message(self, entry: dict) -> str:
        """Format transaction preview for Telegram message."""
        msg = (
            f"NEW MINT APPROVAL REQUEST\n"
            f"{'='*30}\n"
            f"ID: #{entry.get('id', 'N/A')}\n"
            f"Contract: {entry.get('contract_address', 'N/A')}\n"
            f"Chain: {entry.get('chain', 'N/A')}\n"
            f"Wallet: {entry.get('wallet_label', 'N/A')}\n"
            f"Function: {entry.get('mint_function', 'N/A')}\n"
            f"Quantity: {entry.get('quantity', 'N/A')}\n"
            f"Value: {entry.get('total_value_wei', '0')} wei\n"
            f"{'='*30}\n"
            f"\nApprove or Reject?"
        )
        return msg
    
    def send_approval_request(self, entry: dict):
        """Send approval request to allowed users (placeholder)."""
        if not self.token:
            print("  [TELEGRAM] Bot token not configured")
            return
        
        msg = self.format_approval_message(entry)
        print(f"  [TELEGRAM] Would send to {len(self.allowed_users)} user(s):")
        print(f"  {msg[:100]}...")
        
        # Future: Use telegram API to send message with inline keyboard
        # keyboard = InlineKeyboardMarkup([
        #     [InlineKeyboardButton("Approve", callback_data=f"approve_{entry['id']}"),
        #      InlineKeyboardButton("Reject", callback_data=f"reject_{entry['id']}")]
        # ])
    
    def handle_callback(self, user_id: int, action: str, entry_id: int) -> str:
        """Handle approve/reject callback (placeholder)."""
        if not self.is_authorized(user_id):
            return "Unauthorized user"
        
        from custom_tools.approval_queue import approve, reject
        
        if action == "approve":
            approve(entry_id, approved_by=f"telegram:{user_id}")
            return f"Entry #{entry_id} APPROVED"
        elif action == "reject":
            reject(entry_id, reason=f"Rejected by telegram:{user_id}")
            return f"Entry #{entry_id} REJECTED"
        else:
            return "Unknown action"
    
    def start(self):
        """Start the bot (placeholder)."""
        if not self.token:
            print("  ERROR: TELEGRAM_BOT_TOKEN not set")
            print("  Set it in .env file and restart")
            return
        
        if not self.allowed_users:
            print("  ERROR: TELEGRAM_ALLOWED_USERS not set")
            print("  Add comma-separated Telegram user IDs to .env")
            return
        
        print(f"  [TELEGRAM] Bot starting (placeholder)")
        print(f"  [TELEGRAM] Allowed users: {self.allowed_users}")
        print(f"  [TELEGRAM] Full implementation coming soon...")
        
        # Future: Start polling or webhook
        # application = Application.builder().token(self.token).build()
        # application.run_polling()


if __name__ == "__main__":
    bot = TelegramApprovalBot()
    bot.start()
