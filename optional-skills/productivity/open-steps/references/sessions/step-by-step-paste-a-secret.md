# Example 1 - asking the user to place a database password on two servers

Source: a real session message the user could not act on. Translated and
anonymised. This example is why rules 1, 3 and 8 exist.

## BEFORE - unreadable, though everything is technically present

The message opened with three paragraphs of what the agent had already tried:
a cloud CLI path that returned 403, a missing local database client, a second
attempt at the same workaround. Then, under a heading reading "What I need from
you - one command per host", it gave two nearly identical commands with
different IP addresses, separated by a sentence naming which database each one
wanted. It closed with a five-item plan mixing the agent's own next steps with
one item the user had to do, named mid-sentence.

**Defects**

| Defect | How it showed up |
|---|---|
| Ask buried behind diagnostics | Three paragraphs of failures before the request |
| Heading contradicts content | "one command per host" above two commands |
| Jargon inside the instruction | connection string, `sslmode`, Ctrl-D, `umask 077` |
| Two look-alike commands, unlabelled | Which address is the test server is only inferable from prose |
| No verification | Nothing said what success looks like |
| Agent's work mixed with the user's | "this one's on you" hidden inside a list item |

The fourth defect is not style. Swapping the order sends a live-environment
password to the test server.

## AFTER - step-by-step mode output

> **What I need from you: put the database password on two servers.**
>
> Why you and not me: it is a password. Passing through me, it would stay in the
> chat history and in the logs. And I have no database access - my key cannot
> see them.
>
> **Step 1. Get the first password.**
> In the hosting console, open Databases → the test database → Connection
> string. Copy the whole thing; it starts with `postgresql://`.
>
> **Step 2. Give it to the TEST server.**
> Paste this one line and press Enter - it will ask for the value, hide your
> typing, and confirm it saved:
> ```bash
> printf 'Paste the TEST connection string, then press Enter: '; IFS= read -rs V; echo; if [ ${#V} -lt 20 ]; then echo "Only ${#V} characters - looks truncated, nothing saved."; else printf '%s' "$V" | ssh root@<test-server> 'umask 077 && cat > /root/db_admin_dsn' && ssh root@<test-server> 'echo "Saved, $(wc -c < /root/db_admin_dsn) bytes"'; fi; unset V
> ```
>
> **Step 3. Repeat for the LIVE server.**
> A different password - take it from the live database. Do not mix them up:
> this one serves real users.
> ```bash
> printf 'Paste the LIVE connection string, then press Enter: '; IFS= read -rs V; echo; if [ ${#V} -lt 20 ]; then echo "Only ${#V} characters - looks truncated, nothing saved."; else printf '%s' "$V" | ssh root@<live-server> 'umask 077 && cat > /root/db_admin_dsn' && ssh root@<live-server> 'echo "Saved, $(wc -c < /root/db_admin_dsn) bytes"'; fi; unset V
> ```
>
> **How you'll know it worked.**
> Each command ends with "Saved, N bytes". "Only N characters" means the paste
> came in truncated - run it again. `Permission denied` - tell me.
>
> **What happens next, on my side.**
> Say "done with the passwords" - I grant the permissions, check that writing
> works and deleting is refused, then erase both password files.

## What this example fixed in the skill

1. Rule 1 - the ask moved to the top; the failed attempts were dropped entirely.
   They were never information the user could use.
2. Rule 3 - each command now names its target in capital letters, with the
   consequence of swapping them stated.
3. Rule 6 - "Saved, N bytes" is the check. Without it the user cannot tell a
   successful step from a hung one.
4. The secrets section - "it would stay in the chat history and in the logs" is
   the human version of "it does not pass through my context".
5. Rule 4 and the one-line commands - the first version of this very example
   told the user to press Ctrl-D after `cat >`, exactly what the rule now
   forbids. The command asks for the value itself and confirms by byte count.
