/**
 * Normalizes stored `inboxId` into an RFC5322 mailbox address for `$INBOX`
 * substitution (`From`, submission `envelope`, etc.).
 *
 * Credentials may store only the local-part (`alice`); production mailboxes
 * live at `alice@atomicmail.ai`. Pass `ATOMIC_MAIL_INBOX_DOMAIN` via `env`
 * when not using the default domain.
 */
const DEFAULT_INBOX_DOMAIN = "atomicmail.ai";
export function inboxIdToMailboxEmail(inboxId, env) {
    const trimmed = inboxId.trim();
    if (trimmed.length === 0)
        return inboxId;
    if (trimmed.includes("@"))
        return trimmed;
    const raw = env?.ATOMIC_MAIL_INBOX_DOMAIN?.trim();
    const domain = raw && raw.length > 0
        ? raw.replace(/^@+/, "")
        : DEFAULT_INBOX_DOMAIN;
    return `${trimmed}@${domain}`;
}
