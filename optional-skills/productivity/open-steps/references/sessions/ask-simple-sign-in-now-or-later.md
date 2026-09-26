# Example 1 - three questions in one, none of them answerable

Source: a real session in my own analytics dashboard project, condensed. No
client or third-party material.

## What the agent asked

> Should I integrate Clerk with Next.js middleware for route-level protection,
> or defer auth and keep the deployment protection for now? If Clerk, do you
> want the hosted sign-in page or embedded components, and should I model
> organizations from the start in case you go multi-tenant later?

Three questions stacked in one message, each assuming the answer to the one
before it, all in the vendor's vocabulary.

## `ask-simple` mode

> Should the dashboard get real sign-in now, or stay behind the single
> password it has today?
>
> Why it matters: today one password guards everything; real sign-in gives
> each person their own access, which you can take away one at a time.
> What changes later: nothing gets harder by waiting - the sign-in work is the
> same size on the day a second person needs it.
> Easy to undo: yes - the gate is one file and one screen; take them out and
> the password is back.
>
> | Check | Answer |
> |---|---|
> | How long now | Roughly half a day to a day; the gate in front of every page, plus the sign-in screen in the app shell |
> | Simpler substitute | The password already there - it does this one job today |
> | Extra work for you later | Inviting and removing people by hand in the provider's dashboard |
> | Harder to change later | Accounts lock to the provider's user ids; switching providers means re-mapping them |
> | Over-engineering | Yes, today - one person uses this dashboard |
> | Easy to undo | Yes: the gate is one file and one screen; remove them and the password is back |
>
> My recommendation: change nothing until a second person needs access. The
> password does the one job sign-in would do, and the half-day is better spent
> on the dashboard itself.
>
> **Sign-in?**
> - Keep the password (Recommended) - nothing to build; add sign-in the day a second person needs in
> - Add real sign-in now - roughly half a day to a day, then inviting people by hand

## What the rewrite did

| Move | Where |
|---|---|
| Three questions became one | hosted-vs-embedded and multi-tenant only exist after a yes - asking them first buys careless answers (rule 5) |
| Vendor words became what the reader would see | "Clerk with middleware" → "real sign-in"; "deployment protection" → "the single password" |
| Doing nothing was weighed, and won | rule 1 - and the one line says why it won |
| The screen said over-engineering, plainly | one person today; the row exists to be answered yes sometimes |
| Recommended against the more interesting build | the integration is the fun option; that was a reason for suspicion, not preference (rule 6) |
| The picker carried the trade-offs | two options, one line each, the recommended one first and marked |
