# Long WhatsApp Messages — HTML Card Fallback + Unknown-Phone Pattern

## HTML Card Fallback (when a deep-link URL would split)

When the encoded WhatsApp URL exceeds ~4,096 chars in plain text, Telegram splits it into multiple
messages, corrupting the link. User complaint that triggered this: "the whatsapp encoded link got
split across two messages so it won't work."

**Resolution:** HTML card with URL inside an `<a href>` — URL never appears as plain text in the
Telegram message, so Telegram cannot split it.

**Delivery (MANDATORY — DO NOT SKIP):** Send the HTML file to the user **directly via Telegram**
using `send_message` with `MEDIA:` path. Do NOT just save it to `/tmp/` and tell the user it's
there — the user is on Telegram and cannot access the bot's filesystem.

✅ **Correct pattern** (verified working Jun 2026):
```
write_file(path="/tmp/whatsapp-{contact}-{topic}.html", content=html_body)
send_message(message="Here's your file — MEDIA:/tmp/whatsapp-{contact}-{topic}.html")
```

Any writable path works (including `/tmp/`) as long as you send via MEDIA immediately after. The
`/data/hermes/cron/output/` path is also fine for persistent copies. The MEDIA delivery is the
critical step — not the save location.

**Trigger proactively:** If message body is 500+ words, 4+ substantive paragraphs, or multi-section
legal/financial content, generate HTML card immediately. Do not generate plain deep link first and
wait to be told it got split.

**Multi-reference WhatsApp messages** — When the message needs links to multiple documents (sheet +
layout PDF + brochure + masterplan + plot list), always generate an HTML card. The cumulative
URL-encoded text is too long for a single WhatsApp URL, and Telegram will split it. The HTML card
keeps the user in control: they open it in their browser, review the full message, and tap a single
green button that opens WhatsApp with everything pre-filled.

**⚠️ /tmp accessibility pitfall:** The user cannot access `/tmp/` from Telegram. Saving an HTML file
there without also sending it via MEDIA is equivalent to not delivering it at all. Always pair
file-creation with MEDIA delivery.

**⚠️ Container directory pitfall (Jun 2026):** The `/data/hermes/cron/output/` directory may not
exist on all setups. Before writing there, ensure the path exists or fall back to `/tmp/` + MEDIA
delivery.

## HTML Card — Unknown Phone Number Pattern (Jun 2026)

**Trigger:** The user wants a WhatsApp link to a contact whose phone number you cannot find (not in
contacts sheet, People API, memory, past sessions, or any Drive document).

**Problem:** Without a phone number, `wa.me` or `api.whatsapp.com` links cannot pre-select a
recipient. The user has to manually choose the contact when WhatsApp opens. This works fine for
short links but fails for long/multi-reference messages where the URL gets split or truncated.

**Resolution:** Generate an interactive HTML card with:
1. A **phone number input field** (with label + placeholder)
2. A **green "Send via WhatsApp" button** whose `href` updates live as the user types
3. A **message preview box** so the user can see what will be sent
4. Drive links or other reference links listed below the preview
5. JavaScript that builds the `api.whatsapp.com/send?phone=...&text=...` URL dynamically with proper encoding

**HTML structure template:**
```html
<div class="card">
  <h2>Contact Name — Project/Role</h2>
  <div class="phone-row">
    <input type="tel" id="phoneInput" placeholder="e.g. 9199xxxxxxxx">
  </div>
  <a id="waBtn" class="btn" href="#" target="_blank">📤 Send via WhatsApp</a>
  <div class="preview">Message preview...</div>
  <div class="links">
    <strong>DRIVE LINKS:</strong>
    <a href="...">Document 1</a>
    <a href="...">Document 2</a>
  </div>
</div>
<script>
  var message = "...";
  document.getElementById('phoneInput').addEventListener('input', function() {
    var raw = this.value.replace(/[^0-9]/g, '');
    document.getElementById('waBtn').href = 'https://api.whatsapp.com/send?phone=' + raw + '&text=' + encodeURIComponent(message);
  });
</script>
```

**Key rules:**
- NO ampersands (`&`) in the message text — use "and" instead, or the full-width ampersand workaround
- Keep the message concise (under 400 chars) so the wa.me link doesn't get truncated
- Use `api.whatsapp.com/send` NOT `wa.me` — `wa.me` double-decodes query strings
- JavaScript's `encodeURIComponent` handles URL-encoding — no manual encoding needed
- Deliver via `send_message(message="MEDIA:/tmp/whatsapp-{contact}.html")` immediately