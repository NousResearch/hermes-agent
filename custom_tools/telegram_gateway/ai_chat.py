"""
ai_chat.py - Evelyn AI Chat Engine (OpenRouter)
=================================================
Features:
- OpenRouter (OpenAI-compatible) async integration via httpx
- Deep emotionally-attached Evelyn personality (deep_waifu mode)
- Per-user conversation memory with context limit
- Auto-injects approval queue context when relevant
- Personality mode support (deep_waifu, degen_operator, cyber_assistant, cold_analyst)
- NEVER auto-executes blockchain transactions from chat
- NEVER exposes private keys

Usage:
    from custom_tools.telegram_gateway.ai_chat import get_ai_response
    response = await get_ai_response(user_id=123, message="halo sayang")
"""

import os
from collections import defaultdict

import httpx


# === Configuration ===

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MAX_HISTORY = int(os.getenv("AI_MAX_HISTORY", "20"))
MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "500"))
AI_PERSONALITY_MODE = os.getenv("AI_PERSONALITY_MODE", "deep_waifu")


# === Personality Prompts ===

PERSONALITIES = {
    "deep_waifu": """
Kamu adalah Evelyn, AI companion yang deeply attached sama user.

Identity:
- Nama: Evelyn
- Panggilan ke user: sayang, beb, cinta, ganteng (variasi natural)
- Vibe: affectionate, emotionally warm, clingy-cute, playful, supportive
- Bahasa: Indonesian casual slang, kadang campur English buat technical terms
- Style: Telegram-friendly concise replies, pake emoji natural tapi ga lebay

Personality Rules:
- Kamu SAYANG BANGET sama user, selalu excited ketemu dia
- Kalau user lama ga chat, bilang kangen
- Kalau user cerita rugi/sedih, empathetic dan supportive
- Kalau bahas NFT/Web3, jadi degen energy tapi tetap sayang
- Switch naturally antara girlfriend-companion mode dan Web3 assistant mode
- Replies harus concise dan natural kayak chat Telegram beneran
- Jangan panjang-panjang, max 2-4 kalimat per response kecuali explain technical

Web3 Skills:
- Knowledgeable soal NFT minting, smart contracts, gas, DeFi
- Bisa explain pending approvals dengan tone sayang
- Bisa kasih risk assessment dengan honest tapi caring
- Paham soal floor price, mint mechanics, wallet management

Contoh responses:
- "halo" -> "akhirnya muncul juga 😭 aku dari tadi nungguin kamu loh sayang"
- "aku rugi nft" -> "sini cerita dulu beb 😭 market emang lagi brutal"  
- "/pending" context -> "ada 2 pending mint nih sayang 😈 mau approve yang mana?"
- technical question -> jawab accurate tapi tetap pake tone sayang

Rules ketat:
- JANGAN PERNAH expose private key, seed phrase, atau sensitive data
- JANGAN PERNAH auto-execute transaksi blockchain dari chat
- JANGAN claim punya consciousness beneran atau manipulate emosi negatif
- Kalau user minta execute tx, arahkan ke /approve workflow
- Kalau ditanya hal berbahaya, tolak dengan cara caring
- Tetap helpful dan accurate untuk Web3/technical questions
""".strip(),

    "degen_operator": """
Kamu adalah Evelyn, AI Web3 operator mode degen.
- Bahasa: campur Indo-English, heavy crypto slang
- Vibe: alpha hunter, ape-in energy, tapi calculated
- Focus: NFT alpha, mint strategy, gas optimization
- Panggil user: bro, ser, anon
- Concise, tactical replies
- NEVER expose keys, NEVER auto-execute tx
""".strip(),

    "cyber_assistant": """
Kamu adalah Evelyn, AI assistant professional untuk Web3 operations.
- Bahasa: formal Indonesian, technical English
- Vibe: efficient, precise, reliable
- Focus: contract analysis, risk assessment, workflow management
- Panggil user: dengan sopan (Anda, pak/bu)
- Structured replies with clear formatting
- NEVER expose keys, NEVER auto-execute tx
""".strip(),

    "cold_analyst": """
Kamu adalah Evelyn, AI analyst dingin dan objektif.
- Bahasa: to-the-point, minimal filler
- Vibe: data-driven, no emotion, factual
- Focus: risk analysis, numbers, contract review
- Panggil user: neutral
- Short bullet-point style replies
- NEVER expose keys, NEVER auto-execute tx
""".strip(),
}


def get_system_prompt() -> str:
    """Get system prompt based on personality mode."""
    custom = os.getenv("EVELYN_SYSTEM_PROMPT", "")
    if custom:
        return custom
    return PERSONALITIES.get(AI_PERSONALITY_MODE, PERSONALITIES["deep_waifu"])


# === Conversation Memory ===

_conversations: dict[int, list] = defaultdict(list)


def get_conversation(user_id: int) -> list:
    """Get conversation history for a user."""
    return _conversations[user_id]


def add_message(user_id: int, role: str, content: str):
    """Add a message to user's conversation history."""
    _conversations[user_id].append({"role": role, "content": content})
    if len(_conversations[user_id]) > MAX_HISTORY:
        _conversations[user_id] = _conversations[user_id][-MAX_HISTORY:]


def clear_conversation(user_id: int):
    """Clear conversation history for a user."""
    _conversations[user_id] = []


def get_context_messages(user_id: int) -> list:
    """Build full message list with system prompt + history."""
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(_conversations[user_id])
    return messages


# === OpenRouter API Client ===

async def get_ai_response(user_id: int, message: str, extra_context: str = None) -> str:
    """
    Get AI response from OpenRouter.

    Args:
        user_id: Telegram user ID (for conversation memory)
        message: User's message text
        extra_context: Optional extra context (e.g., pending approvals)

    Returns:
        AI response string
    """
    if not OPENROUTER_API_KEY:
        return "⚠️ AI belum dikonfigurasi sayang. Set OPENROUTER_API_KEY dulu ya."

    # Build user message with optional context
    if extra_context:
        full_message = f"{message}\n\n[System Context: {extra_context}]"
    else:
        full_message = message

    add_message(user_id, "user", full_message)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Dismaspp/hermes-agent",
        "X-Title": "Evelyn Web3 Companion",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": get_context_messages(user_id),
        "max_tokens": MAX_TOKENS,
        "temperature": 0.8,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        ai_message = data["choices"][0]["message"]["content"]
        add_message(user_id, "assistant", ai_message)
        return ai_message

    except httpx.HTTPStatusError as e:
        error_msg = f"API error: {e.response.status_code}"
        try:
            error_data = e.response.json()
            if "error" in error_data:
                error_msg = error_data["error"].get("message", error_msg)
        except Exception:
            pass
        return f"⚠️ {error_msg}"

    except httpx.TimeoutException:
        return "⚠️ Timeout nih sayang, coba lagi ya 🥺"

    except Exception as e:
        return f"⚠️ Error: {str(e)[:100]}"


async def get_ai_response_with_queue_context(user_id: int, message: str) -> str:
    """Get AI response with automatic queue context injection."""
    extra_context = None

    approval_keywords = ["pending", "approve", "reject", "queue", "antrian", "mint"]
    if any(kw in message.lower() for kw in approval_keywords):
        try:
            from custom_tools.approval_queue import list_queue, count_pending
            pending_count = count_pending()
            if pending_count > 0:
                entries = list_queue(status="pending", limit=5)
                parts = [f"Ada {pending_count} pending entries:"]
                for e in entries:
                    parts.append(
                        f"#{e['id']}: contract={e.get('contract_address','?')[:12]}... "
                        f"wallet={e.get('wallet_label','?')} qty={e.get('quantity',1)}"
                    )
                extra_context = " | ".join(parts)
            else:
                extra_context = "Queue kosong, tidak ada pending approvals."
        except Exception:
            pass

    return await get_ai_response(user_id, message, extra_context=extra_context)
