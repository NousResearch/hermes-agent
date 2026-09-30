# System prompt

Keep native Hermes identity/SOUL, memory guidance, operational guidance,
project/environment context, provider instructions and cache lifecycle.
The main prompt has four deliberate differences:

- Native Hermes help wording points to the shipped guide through `read_file`.
- A short connection pointer requires reading and maintaining service manuals.
- The existing full responsibility renderer occupies the native skills-index
  position. Ownership, state, correction rules, roster and warnings are unchanged.
- Shared memory remains in system context; global USER.md is omitted. Personal
  memory follows the current user message before recall.

Do not add the source product's Memory section or a separate background-memory
paragraph. Hindsight's tool description provides its usage guidance. The memory
tool uses the source product's mechanics-only description, with personal and
shared organization targets; native store operations and flags remain.

`agent/system_prompt.py` owns assembly. `agent/employee_prompt.py` supplies the
connection pointer and responsibility roster. These are frozen with the native
conversation prompt, including across warm turns. Existing responsibility and
connection paths remain provisional pending the folder decision.
