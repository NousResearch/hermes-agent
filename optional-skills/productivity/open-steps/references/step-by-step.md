# step-by-step mode

The user is not stuck because the task is hard - they cannot tell what they
are being asked to do. This skill turns "I need something from you" into an
instruction a person who does not read code can follow without a follow-up
question. `done-or-not` mode reports *that* something is needed; this one says
exactly *how*.

## Language

Write in the language the user speaks in this session, detected from the
conversation. Commands, file names and identifiers stay English.

## When to use

Triggers live in the description above - any moment you need the user's hands,
or they ask what to do.

## Step 0 - earn the right to ask

Every ask costs the user a context switch. Prove it is necessary; stop at the
first item that clears the block:

1. **Try it.** A real 403 is a finding; "I probably lack permission" is a guess.
2. **Find another route.** Another tool, a value already on the host, a file
   you can read.
3. **Shrink the ask.** Obtain what you can yourself; hand over only the
   irreducible part.
4. **Check you are not asking twice** - search the session and the reports
   folder first.

Three walls where asking IS the correct move, never to be worked around:
pulling a secret into your own context, loosening a guard that is there on
purpose, doing what the user said only they do.

## The shape

Always this order - the ask first, never after the diagnosis.

```
**What I need from you: <one sentence, plain words>.**

Why you and not me: <one or two sentences. A real reason, in human terms.>

**Step 1. <action in three to six words>**
<What to do. One action only.>

**Step 2. <action>**
<...>

**How you'll know it worked.**
<What the user will see. What to do if they see something else.>

**What happens next, on my side.**
<One line: what you do once they are done, and what you will say.>
```

## After they act - verify, do not trust

"Done" is a claim. Run the one quickest check that would fail if the step had
not worked - the file exists with the right owner, the service answers, the
value works once. Never print a secret to confirm it: confirm its effect. If
the check fails, give only the corrected step - never the whole list again.
Verified versus assumed is exactly what `done-or-not` mode needs for "yes"
versus "not checked".

## Secrets and dangerous steps

A secret never goes through the chat - it would stay in the history and the
logs. The command below puts it where it belongs directly; say who deletes it
and when. A hard-to-undo step - live users, money, deletion - gets its own
warning line before the command.

**Writing a secret to a remote host** - one line: it prompts, hides the
typing, refuses a truncated paste, confirms by size:

```bash
printf 'Paste the connection string, then press Enter: '; IFS= read -rs V; echo; if [ ${#V} -lt 20 ]; then echo "Only ${#V} characters - that looks truncated, nothing was saved."; else printf '%s' "$V" | ssh root@HOST 'umask 077 && cat > /path/to/secret' && ssh root@HOST 'echo "Saved, $(wc -c < /path/to/secret) bytes"'; fi; unset V
```

**Typed by hand rather than pasted** - ask twice; a typo in a hidden field is
otherwise undetectable:

```bash
printf 'Enter the token: '; IFS= read -rs A; echo; printf 'Enter it again: '; IFS= read -rs B; echo; if [ "$A" != "$B" ]; then echo "The two entries differ - nothing saved, run it again."; else printf '%s' "$A" | (umask 077; cat > /path/to/secret) && echo "Saved, $(wc -c < /path/to/secret) bytes"; fi; unset A B
```

Two properties the templates cannot keep for you, both measured:

- **`printf …; IFS= read -rs VAR` - never `read -rsp`.** In zsh `-p` means
  "read from a coprocess": the variable comes back empty with no error and the
  secret file is written blank. macOS defaults to zsh.
- **The value never appears in the command itself** - only piped from the
  variable; anything in the arguments lands in shell history and the process
  list.

The rest the templates already embody - one single line, `umask 077` before
writing, refuse short input, confirm by byte count never by content, `unset`
at the end. Adapt the prompt, the threshold and the path; keep every property.

**The same pattern serves any value the user must supply by hand** - a public
key, a domain, an address, an id. Prompt for it the same way; keep the typing
visible when the value is not secret (drop `-s`), skip the length gate when
short is valid - and always end with a plain-words confirmation of what just
happened, so pressing Enter never feels like dropping a coin into a well.

## Choices, not instructions

A *decision* gets no steps. Use a numbered option list (or the platform's option picker where one exists) with the pack's contract:
plain question, why it matters, what changes later, easy to undo, two to four
options, the recommended one first and marked. On a chat surface without a picker, a numbered list is the same content.

## Hard rules

1. **The ask goes first.** What you tried and what failed is your problem -
   one line at the end, or nothing.
2. **One action per step.** Two commands is two steps.
3. **Label every command with what it touches** - the test server, the live
   server, their own machine. Look-alike steps on different targets: say what
   happens if they are swapped.
4. **A command is self-contained.** One line the user pastes and runs; if it
   needs a value, it asks for it. Never input redirection, heredocs, Ctrl-D.
5. **No jargon inside a step.** Avoid terms instead of explaining them: write
   what the person sees and clicks. One unavoidable term may stay - without a
   lecture.
6. **Always give a way to check.** A step the user cannot verify is a step
   they will redo out of doubt.
7. **Never mix your work with theirs.** Two lists with plain headings; a
   buried "this one's on you" is not an instruction.
8. **Never compress a multi-step sequence** - here brevity causes misreads.
   Blocks of two to three sentences; longer gets skimmed, and a skimmed step
   is a missed step.

## Known gotchas

- A heading is not a summary: "one command per host" above two commands reads
  as one command. Count out loud.
- A dropped step is "skip step 3" - never a silent renumber.
- If you can verify it yourself, verify it yourself; do not ask for
  confirmation you do not need.
- "Say done" needs a subject - the user may have three of your requests open.
