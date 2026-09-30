# Handoff message

`scripts/host_facts.py` sends these sections with the fork card's result. Each `## ` heading names one section.

## message

The handoff message is the new chat's first user message. It is visible, so write it as their own ask, in their language, in the first person. The app adds the rest under it: a folded "What setup learned about me" block (their name, app, plugin and layout picks, and what the scan saw) and the first-task rules, which connect the apps and install the plugins they picked before the task starts. Do not repeat any of that, and add nothing they did not say.

Write these parts in order, one short paragraph each (part names are not text). Drop a part whose slot is empty.

1. The ask, in one or two sentences, in their words where they gave them, keeping the named app and the outcome. A fork row they picked is the ask as its label says it ("A daily brief from Linear and Slack."). "I have something in mind" with nothing typed stays as it is. No machine specs.
2. What they are working on, if they said it, and any plugin they asked for after the plugins card.
3. The plan below, word for word.

## build

Keep the first pass small: I would like to see something done in a few minutes.

## machine

Help me set up this <machine_kind> (<description>). I mainly use it for <their answer>. Start with a few apps I'll want.
