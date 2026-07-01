/** Static JMAP success hints (no shared/ filesystem reads). */
export const JMAP_NEXT_HINTS = [
    "Use jmap_request with Mailbox/get or Email/query to work with mail data.",
    "Use presets with $VAR placeholders — $ACCOUNT_ID, $INBOX, and $INBOX_MAILBOX_ID come from the session; pass others via vars / --vars.",
    "Call help for the JMAP cheatsheet and troubleshooting.",
];
