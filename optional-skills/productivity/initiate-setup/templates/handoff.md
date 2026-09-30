# Handoff message

`scripts/host_facts.py` sends these sections with the fork card's result, the machine plan cut to this computer. Each `## ` heading names one section.

## message

The handoff message is the new chat's first user message. It is visible, so write it as their own ask, in their language, in the first person. Nothing else reaches the task chat: no memory, no hidden note.

The install list: the plugins they picked that the task needs (all of them for machine setup or a task naming the app), plus a picked first task's own `plugins`. The connect list: apps the task needs that setup did not connect; none for machine setup.

Write these parts in order, one paragraph each (part names are not text). Copy the quoted wording, fill only the `<slots>`, and drop a sentence, clause or part whose slot is empty or whose condition does not hold. Add nothing they did not say. Check every non-empty part is there.

1. The ask, in one or two sentences, in their words where they gave them, keeping the named app and the outcome. No machine specs.
2. "Call me <name>." Then what they are working on, if they said it.
3. "I use: <all app picks, exact ids>. Connected during setup: <apps setup connected>."
4. Only with an install or connect list: "In your first turn, before anything else: install <install list> with one `manage_catalog` install call carrying all of them, then connect <connect list> with one `manage_connections` connect call carrying all of them. The ids are exact; skip search and status checks, and start the work when they return. If I skip some, go on without them and tell me in one line what each would have added; do not offer them again." Drop the install or the connect clause when its list is empty. With an install list, add: "Find plugin tools with `tool_search` and read a plugin's skill with `skill_view` by its exact name; if its app is not running, tell me plainly." With other plugin picks, add: "Picked during setup, not needed yet: <other plugin picks>."
5. The plan below, word for word.
6. Last (drop the first sentence when `scan.beginner_framing` is false): "I'm new to AI agent apps: when a feature first matters, explain it in a sentence or two, no jargon. As you start, tell me in one short sentence that you'll ask for permissions as you go and I can say no or redirect you. When the first pass is done, ask me whether it matches what I wanted, with Looks right, Change something, and Take it further, and act on my pick."

## build

"Plan briefly, then build: scaffold, research, first artifact. Use real data only from connected apps and find their tools with `tool_search`; tools already signed in on this computer, like a logged-in gh, are fair to use, and say so in one line. Never route around a connector: no IMAP client, app password or other way into the same account; if I decline one, that app is out of this build. Ask before sending, deleting or scheduling anything, and set up no recurring job unless I asked. Make the result something I can open: one HTML page if the idea allows it." With a connect list, add: "Include at least one real reading or action through a connected app." Without one, add: "Make this first version finishable with no account I have not connected: web research with the browser visible to me, scripts, a small app, a file-based tracker, a generated page. If the idea needs an account, build the no-account core first and offer the connection next."

## machine

Get this computer ready to use, end to end, with the terminal. It needs no account; never send me to a sign-in for it. Setup signals, not proof of the machine's age: <description>. I mainly use it for <their answer>. Look first: OS and version, architecture, pending updates, free disk, the package manager, and which everyday tools are installed (browser, editor, git, python, node, docker, the apps I named).

## machine-nvidia

Also check the GPU and driver (nvidia-smi), a container runtime and the CUDA toolchain.

## machine-plan

Tell me what you found in a few plain lines. Match the plan to my use: email, calendars, documents and meetings need no developer stack. Recommend WSL or CUDA only as a verified need of my use, benefit first; never WSL on Linux or macOS, never a CUDA reinstall just because this is a Spark. Propose a short numbered plan, most useful first: updates, a package manager if missing, my everyday tools, sane defaults, then anything exotic. Ask before running it, with Go ahead, Change the list, and Just the essentials. Then one step at a time, one line per step on what it is for. Prefer native, already-working tools and the official package manager over downloaded installers. Never install what I did not agree to, overwrite config without asking, or disable security settings; stop and ask when anything looks destructive or wants a password I did not give.

## machine-drivers-win32

Drivers: check for missing or unknown devices and vendor GPU drivers, and say so when Windows already handles it.

## machine-drivers-darwin

Drivers: system updates and the App Store cover drivers on macOS, so say that.

## machine-drivers-linux

Drivers: check the kernel and driver pairing before touching the GPU.

## machine-arm

On Arm, check the architecture for every install, prefer native arm64 builds, say when only an emulated x64 one exists, and never assume a tool has an Arm release.

## machine-arm-nvidia-win32

On an Arm Windows PC with NVIDIA silicon, treat CUDA and anything GPU-related as arm64-specific and verify the build first.

## machine-end

Anything needing my sign-in, a licence key or a payment goes on a list for me. Finish with what changed, what you skipped and why, what is left for me, and whether a reboot is needed.

## machine-crashes

It shut down unexpectedly <crash_30d> times in the last 30 days; find out why as part of the look.
