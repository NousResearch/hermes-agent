# twozero MCP Tools Reference

36 tools from twozero MCP v2.774+ (April 2026).
All tools accept an optional `target_instance` param for multi-TD-instance scenarios.

## Execution & Scripting

### td_execute_python

Execute Python code inside TouchDesigner and return the result. Has full access to TD Python API (op, project, app, etc). Print statements and the last expression value are captured. Best for: wiring connections (inputConnectors), setting expressions (par.X.expr/mode), querying parameter names, and batch creation scripts (5+ operators). For creating 1-4 operators, prefer td_create_operator instead.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | yes | Python code to execute in TouchDesigner |

## Network & Structure

### td_get_network

Get the operator network structure in TouchDesigner (TD) at a given path. Returns compact list: name OPType flags. First line is full path of queried op. Flags: ch:N=children count, !cook=allowCooking off, bypass, private=isPrivate, blocked:reason, "comment text". depth=0 (default) = current level only. depth=1 = one level of children (indented). To explore deeper, call again on a specific COMP path. System operators (/ui, /sys) are hidden by default.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | no | Network path to inspect, e.g. '/' or '/project1' |
| `depth` | integer | no | How many levels deep to recurse. 0=current level only (recommended), 1=include direct children of COMPs |
| `includeSystem` | boolean | no | Include system operators (/ui, /sys). Default false. |
| `nodeXY` | boolean | no | Include nodeX,nodeY coordinates. Default false. |

### td_create_operator

Create a new operator (node) in TouchDesigner (TD). Preferred way to create operators — handles viewport positioning, viewer flag, and docked ops automatically. For batch creation (5+ ops), you may use td_execute_python with a script instead, but then call td_get_hints('construction') first for correct parameter names and layout rules. Supports all TD operator types: TOP, CHOP, SOP, DAT, COMP, MAT. If parent is omitted, creates in the currently open network at the user's viewport position. When building a container: first create baseCOMP (no parent), then create children with parent=compPath.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | Operator type, e.g. 'textDAT', 'constantCHOP', 'noiseTOP', 'transformTOP', 'baseCOMP' |
| `parent` | string | no | Path to the parent operator. If omitted, uses the currently open network in TD. |
| `name` | string | no | Name for the new operator (optional, TD auto-names if omitted) |
| `parameters` | object | no | Key-value pairs of parameters to set on the created operator |

### td_find_op

Find operators by name and/or type across the project. Returns TSV: path, OPType, flags. Flags: bypass, !cook, private, blocked:reason. Use td_search to search inside code/expressions; use td_find_op to find operators themselves.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | no | Substring to match in operator name (case-insensitive). E.g. 'noise' finds noise1, noise2, myNoise. |
| `type` | string | no | Substring to match in OPType (case-insensitive). E.g. 'noiseTOP', 'baseCOMP', 'CHOP'. Use exact type for precision or partial for broader matches. |
| `root` | string | no | Root operator path to search from. Default '/project1'. |
| `max_results` | number | no | Maximum results to return. Default 50. |
| `max_depth` | number | no | Max recursion depth from root. Default unlimited. |
| `detail` | `basic` / `summary` | no | Result detail level. 'basic' = name/path/type (fast). 'summary' = + connections, non-default pars, expressions. Default 'basic'. |

### td_search

Search for text across all code (DAT scripts), parameter expressions, and string parameter values in the TD project. Returns TSV: path, kind (code/expression/parameter/ref), line, text. JSON when context>0. Words are OR-matched. Use quotes for exact phrases: 'GetLogin "op('login')"'. Use count_only=true to quickly check if something is referenced without fetching full results.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | Search query. Multiple words = OR (any match). Wrap in quotes for exact phrase. Example: 'GetLogin getLogin' finds either. |
| `root` | string | no | Root operator path to search from. Default '/project1'. |
| `scope` | `all` / `code` / `editable` / `expressions` / `parameters` | no | What to search. 'code' = DAT scripts only (fast, ~0.05s). 'editable' = only editable code (skips inherited/ref DATs). 'expressions' = parameter expressions only. 'parameters' = string parameter values only. 'all' = everything (slow, ~1.5s due to parameter scan). Default 'all'. |
| `case_sensitive` | boolean | no | Case-sensitive matching. Default false. |
| `max_results` | number | no | Maximum results to return. Default 50. |
| `context` | number | no | Lines to show before/after each code match. Saves td_read_dat calls. Default 0. |
| `count_only` | boolean | no | Return only match count, not results. Fast existence check. |
| `max_depth` | number | no | Max recursion depth from root. Default unlimited. |

### td_navigate_to

Navigate the TouchDesigner Network Editor viewport to show a specific operator. Opens the operator's parent network and centers the view on it. Use this to show the user where a problem is, or to navigate to an operator before modifying it.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Path to the operator to navigate to, e.g. '/project1/noise1' |

## Operator Inspection

### td_get_operator_info

Get information about a specific operator (node) in TouchDesigner (TD). detail='summary': connections, non-default pars, expressions, CHOP channels (compact). detail='full': all of the above PLUS every parameter with value/default/label.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Full path to the operator, e.g. '/project1/noise1' |
| `detail` | `summary` / `full` | no | Level of detail. 'summary' = connections, expressions, non-default pars, custom pars (pulse marked), CHOP channels. 'full' = summary + all parameters. Default 'full'. |

### td_get_operators_info

Get information about multiple operators in one call. Returns an array of operator info objects. Use instead of calling td_get_operator_info multiple times.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `paths` | array | yes | Array of full operator paths, e.g. ['/project1/null1', '/project1/null2'] |
| `detail` | `summary` / `full` | no | Level of detail. Default 'summary'. |

### td_get_par_info

Get parameter names and details for a TouchDesigner operator type. Without specific pars: returns compact list of all parameters with their names, types, and menu options. With pars: returns full details (help text, menu values, style) for specific parameters. Use this when you need to know exact parameter names before setting them.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `op_type` | string | yes | TD operator type name, e.g. 'noiseTOP', 'blurTOP', 'lfoCHOP', 'compositeTOP' |
| `pars` | array | no | Optional list of specific parameter names to get full details for |

## Parameter Setting

### td_set_operator_pars

Set parameters and flags on an operator in TouchDesigner (TD). Safer than td_execute_python for simple parameter changes. Can set values, toggle bypass/viewer, without writing Python code.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Path to the operator |
| `parameters` | object | no | Key-value pairs of parameters to set |
| `bypass` | boolean | no | Set bypass state of the operator (not available on COMPs) |
| `viewer` | boolean | no | Set viewer state of the operator |
| `allowCooking` | boolean | no | Set cooking flag on a COMP. When False, internal network stops cooking (0 CPU). COMP-only. |

## Data Read/Write

### td_read_dat

Read the text content of a DAT operator in TouchDesigner (TD). Returns content with line numbers. Use to read scripts, extensions, GLSL shaders, table data.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Path to the DAT operator |
| `start_line` | integer | no | Start line (1-based). Omit to read from beginning. |
| `end_line` | integer | no | End line (inclusive). Omit to read to end. |

### td_write_dat

Write or patch text content of a DAT operator in TouchDesigner (TD). Can do full replacement or StrReplace-style patching (old_text -> new_text). Use for editing scripts, extensions, shaders. Does NOT reinit extensions automatically.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Path to the DAT operator |
| `text` | string | no | Full replacement text. Use this OR old_text+new_text, not both. |
| `old_text` | string | no | Text to find and replace (must be unique in the DAT) |
| `new_text` | string | no | Replacement text |
| `replace_all` | boolean | no | If true, replaces ALL occurrences of old_text (default: false, requires unique match) |

### td_read_chop

Read CHOP channel sample data. Returns channel values as arrays. Use when you need the actual sample values (animation curves, lookup tables, waveforms), not just the summary from td_get_operator_info.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Path to the CHOP operator |
| `channels` | array | no | Channel names to read. Omit to read all channels. |
| `start` | integer | no | Start sample index (0-based). Omit to read from beginning. |
| `end` | integer | no | End sample index (inclusive). Omit to read to end. |

### td_read_textport

Read the last N lines from the TouchDesigner (TD) log/textport (console output). Use this to see errors, warnings and print output from TD.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `lines` | integer | no | Number of recent lines to return |

### td_clear_textport

Clear the MCP textport log buffer. Use this before starting a debug session or an edit-run-check loop to keep td_read_textport output focused and minimal.

No parameters (other than optional `target_instance`).

## Visual Capture

### td_get_screenshot

Get a screenshot of an operator's viewer in TouchDesigner (TD). Saves the image to a file and returns the file path. Use your file-reading tool to view the image. Shows what the operator looks like in its viewer (TOP output, CHOP waveform graph, SOP geometry, DAT table, parameter UI, etc). Use this to visually inspect any operator, or to generate images via TD for use in your project. TWO-STEP ASYNC USAGE: Step 1 — call with 'path' to start: returns {'status': 'pending', 'requestId': '...'}. Step 2 — call with 'request_id' to retrieve: returns {'file': '/tmp/.../opname_id.jpg'}. Then read the file to see the image. If step 2 still returns pending, make one other tool call then retry.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | no | Full operator path to screenshot, e.g. '/project1/noise1'. Required for step 1. |
| `request_id` | string | no | Request ID from step 1 to retrieve the completed screenshot. |
| `max_size` | integer | no | Max pixel size for the longer side (default 512). Use 0 for original operator resolution (useful for pixel-accurate UI work). Higher values (e.g. 1024) for more detail. |
| `output_path` | string | no | Optional absolute path where the image should be saved (e.g. '/Users/me/project/render.png'). If omitted, saved to /tmp/pisang_mcp/screenshots/. Use absolute paths — TD's working directory may differ from the agent's. |
| `as_top` | boolean | no | If true, captures the operator directly as a TOP (bypasses the viewer renderer), preserving alpha/transparency. Only works for TOP operators — if the target is not a TOP, falls back to the viewer automatically. Use this when you need a clean PNG with alpha, e.g. to save a generated image for use in another project. |
| `format` | `auto` / `jpg` / `png` | no | Image format. 'auto' (default): JPEG for viewer mode, PNG for as_top=true. 'jpg': always JPEG (smaller). 'png': always PNG (lossless). |

### td_get_screenshots

Get screenshots of multiple operators in one batch. Saves images to files and returns file paths. Use your file-reading tool to view images. TWO-STEP ASYNC USAGE: Step 1 — call with 'paths' array to start: returns {'status': 'pending', 'batchId': '...', 'total': N}. Step 2 — call with 'batch_id' to retrieve: returns {'files': [{op, file}, ...]}. Then read the files to see the images. If still processing returns {'status': 'pending', 'ready': K, 'total': N}.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `paths` | array | no | List of full operator paths to screenshot. Required for step 1. |
| `batch_id` | string | no | Batch ID from step 1 to retrieve completed screenshots. |
| `max_size` | integer | no | Max pixel size for longer side (default 512). Use 0 for original resolution. |
| `as_top` | boolean | no | If true, captures TOP operators directly (preserves alpha). Non-TOP operators fall back to viewer. |
| `output_dir` | string | no | Optional absolute path to a directory. Each screenshot saved as <opname>.jpg or .png inside it and kept on disk. |
| `format` | `auto` / `jpg` / `png` | no | Image format. 'auto' (default): JPEG for viewer mode, PNG for as_top=true. 'jpg': always JPEG (smaller). 'png': always PNG (lossless). |

### td_get_screen_screenshot

Capture a screenshot of the actual screen via TD's screenGrabTOP. Saves the image to a file and returns the file path. Use your file-reading tool to view the image. Unlike td_get_screenshot (operator viewer), this shows what the user literally sees on their monitor — TD windows, UI panels, everything. Use when simulating mouse/keyboard input to verify what happened on screen. Workflow: td_get_screen_screenshot → read file → td_input_execute → wait idle → td_get_screen_screenshot again. TWO-STEP ASYNC: Step 1 — call without request_id: returns {'status':'pending','requestId':'...'}. Step 2 — call with request_id: returns {'file': '/tmp/.../screen_id.jpg', 'info': '...metadata...'}. Then read the file to see the image. The requestId also stays usable with td_screen_point_to_global for later coordinate lookup. crop_x/y/w/h are in ACTUAL SCREEN PIXELS (not image pixels). Crops exceeding screen bounds are auto-clamped. SMART DEFAULTS: max_size is auto when omitted — 1920 for full screen (good overview), max(crop_w,crop_h) for cropped (guarantees 1:1 scale). At 1:1 scale: screen_coord = crop_origin + image_pixel. Otherwise use the formula from metadata.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | no | Request ID from step 1 to retrieve the completed screenshot. |
| `max_size` | integer | no | Max pixel size for the longer side. Auto when omitted: 1920 for full screen, max(crop_w,crop_h) for cropped (1:1). Set explicitly to override. |
| `crop_x` | integer | no | Left edge in screen pixels. |
| `crop_y` | integer | no | Top edge in screen pixels (y=0 at top of screen). |
| `crop_w` | integer | no | Width in pixels. |
| `crop_h` | integer | no | Height in pixels. |
| `display` | integer | no | Screen index (default 0 = primary display). |

## Context & Focus

### td_get_focus

Get the current user focus in TouchDesigner (TD): which network is open, selected operators, current operator, and rollover (what is under the mouse cursor). IMPORTANT: when the user says 'this operator' or 'вот этот', they mean the SELECTED/CURRENT operator, NOT the rollover. Rollover is just incidental mouse position and should be ignored for intent. Pass screenshots=true to immediately start a screenshot batch for all selected operators — response includes a 'screenshots' field with batchId; retrieve with td_get_screenshots(batch_id=...).

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `screenshots` | boolean | no | If true, start a screenshot batch for all selected operators. Retrieve with td_get_screenshots(batch_id=...). |
| `max_size` | integer | no | Max screenshot size when screenshots=true (default 512). |
| `as_top` | boolean | no | Passed to the screenshot batch when screenshots=true. |

### td_get_errors

Find errors and warnings in TouchDesigner (TD) operators. Checks operator errors, warnings, AND broken parameter expressions (missing channels, bad references, etc). Also includes recent script errors from the log (tracebacks), grouped and deduplicated — e.g. 1000 identical mouse-move errors shown as ×1000 with one entry. If path is given, checks that operator and its children. If no path, checks the currently open network. Use '/' for entire project. Use when user says something is broken, has errors, red nodes, горит ошибка, etc. TIP: call td_clear_textport before reproducing an error to keep log focused. TIP: combine with td_get_perf when user says 'тупит/лагает' to check both errors and performance.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | no | Path to check. If omitted, checks the current network. Use '/' to scan entire project. |
| `recursive` | boolean | no | Check children recursively (default true) |
| `include_log` | boolean | no | Include recent script errors from log, grouped by unique signature (default true). Use td_clear_textport before reproducing an error to keep results focused. |

### td_get_perf

Get performance data from TouchDesigner (TD). Returns TSV: header with fps/budget/memory summary, then slowest operators sorted by cook time. Columns: path, OPType, cpu/cook(ms), gpu/cook(ms), cpu/s, gpu/s, rate, flags. Use when user reports lag, low FPS, slow performance, тупит, тормозит.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | no | Path to profile. If omitted, profiles the current network. Use '/' for entire project. |
| `top` | integer | no | Number of slowest operators to return |

## Documentation

### td_get_docs

Get comprehensive documentation on a TouchDesigner topic. Unlike td_get_hints (compact tips), this returns in-depth reference material. Call without arguments to see available topics with descriptions. Call with a topic name to get the full documentation.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `topic` | string | no | Topic to get docs for. Omit to list available topics. |

### td_get_hints

Get TouchDesigner tips and common patterns for a topic. Call this BEFORE creating operators or writing TD Python code to learn correct parameter names, expressions, and idiomatic approaches. Available topics: animation, noise, connections, parameters, scripting, construction, ui_analysis, panel_layout, screenshots, input_simulation, undo. IMPORTANT: always call with topic='construction' before building multi-operator setups to get correct TOP/CHOP parameter names, compositeTOP input ordering, and layout guidelines. IMPORTANT: always call with topic='input_simulation' before using td_input_execute to learn focus recovery, coordinate systems, and testing workflow.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `topic` | string | yes | Topic to get hints for. Available: 'animation', 'noise', 'connections', 'parameters', 'scripting', 'construction', 'ui_analysis', 'panel_layout', 'screenshots', 'input_simulation', 'undo', 'networking', 'all' |

### td_agents_md

Read, write, or update the agents_md documentation inside a COMP container. agents_md is a Markdown textDAT describing the container's purpose, structure, and conventions. action='read': returns content + staleness check (compares documented children vs live state). action='update': refreshes auto-generated sections (children list, connections) from live state, preserves human-written sections. action='write': sets full content, creates the DAT if missing.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Path to the COMP container |
| `action` | `read` / `update` / `write` | yes | read=get content+staleness, update=refresh auto sections, write=set content |
| `content` | string | no | Markdown content (only for action='write') |

## Input Automation

### td_input_execute

Send a sequence of mouse/keyboard commands to TouchDesigner. Commands execute sequentially with smooth bezier movement. Returns immediately — poll td_input_status() until status='idle' before proceeding. Command types: 'focus' — bring TD to foreground. 'move' — smooth mouse move: {type,x,y,duration,easing}. 'click' — click: {type,x,y,button,hold,duration,easing}. hold=seconds to hold down. duration=smooth move before click. 'dblclick' — double click: {type,x,y,duration}. 'mousedown'/'mouseup' — {type,x,y,button}. 'key' — keystroke: {type,keys} e.g. 'ctrl+z','tab','escape','shift+f5'. Requires Accessibility permission on Mac. 'type' — human-like typing: {type,text,wpm,variance} — layout-independent Unicode, variable timing. 'wait' — pause: {type,duration}. 'scroll' — {type,x,y,dx,dy,steps} — human-like scroll: moves mouse to (x,y) first, then sends dy (vertical, +up) and dx (horizontal, +right) as multiple ticks with natural timing. steps=4 by default. Mouse commands may include coord_space='logical' (default) or coord_space='physical'. On macOS, 'physical' means actual screen pixels from td_get_screen_screenshot and is converted to CGEvent logical coords automatically. Top-level coord_space applies to commands that do not override it. on_error: 'stop' (default) clears queue on error; 'continue' skips failed command. IMPORTANT: call td_get_hints('input_simulation') before first use to learn focus recovery, coordinate systems, and testing workflow.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `commands` | array | yes | List of command dicts to execute in sequence. |
| `coord_space` | `logical` / `physical` | no | Default coordinate space for mouse commands that do not specify their own coord_space. 'logical' uses CGEvent coords directly. 'physical' uses actual screen pixels from td_get_screen_screenshot and is auto-converted on macOS. |
| `on_error` | `stop` / `continue` | no | What to do on error. Default 'stop'. |

### td_input_status

Get current status of the td_input command queue. Poll this after td_input_execute until status='idle'. Returns: status ('idle'/'running'), current command, queue_remaining, last error.

No parameters (other than optional `target_instance`).

### td_input_clear

Clear the td_input command queue and stop current execution immediately.

No parameters (other than optional `target_instance`).

### td_op_screen_rect

Get the screen coordinates of an operator node in the network editor. Returns {x,y,w,h,cx,cy} where cx,cy is the center for clicking. Use this to find where to click on a specific operator. Only works if the operator's parent network is currently open in a network editor pane.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Full path to the operator, e.g. '/project1/myComp/noise1' |

### td_click_screen_point

Resolve a point inside a previous td_get_screen_screenshot result and click it. Pass the screenshot request_id plus either normalized u/v or image_x/image_y. Queues a td_input click using physical screen coordinates, so it works directly with screenshot-derived points. Use duration/easing to control the cursor travel before the click.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | yes | Request ID originally returned by td_get_screen_screenshot. |
| `u` | number | no | Normalized horizontal position inside the screenshot region (0=left, 1=right). Use with v. |
| `v` | number | no | Normalized vertical position inside the screenshot region (0=top, 1=bottom). Use with u. |
| `image_x` | number | no | Horizontal pixel coordinate inside the returned screenshot image. Use with image_y. |
| `image_y` | number | no | Vertical pixel coordinate inside the returned screenshot image. Use with image_x. |
| `button` | `left` / `right` / `middle` | no | Mouse button to click. Default left. |
| `hold` | number | no | Seconds to hold the mouse button down before releasing. |
| `duration` | number | no | Seconds for the cursor to travel to the target before clicking. |
| `easing` | `linear` / `ease-in` / `ease-out` / `ease-in-out` | no | Cursor movement easing for the pre-click travel. |
| `focus` | boolean | no | If true, bring TD to the front before clicking and wait briefly for focus to settle. |

### td_screen_point_to_global

Convert a point inside a previous td_get_screen_screenshot result into absolute screen coordinates. Pass the screenshot request_id plus either normalized u/v (0..1 inside that screenshot region) or image_x/image_y in returned image pixels. Returns absolute physical screen coordinates, logical coordinates, and a ready-to-use td_input_execute payload. Metadata is kept for the most recent screen screenshots so multiple agents can resolve points later by request_id.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | yes | Request ID originally returned by td_get_screen_screenshot. |
| `u` | number | no | Normalized horizontal position inside the screenshot region (0=left, 1=right). Use with v. |
| `v` | number | no | Normalized vertical position inside the screenshot region (0=top, 1=bottom). Use with u. |
| `image_x` | number | no | Horizontal pixel coordinate inside the returned screenshot image. Use with image_y. |
| `image_y` | number | no | Vertical pixel coordinate inside the returned screenshot image. Use with image_x. |

## System

### td_list_instances

List all running TouchDesigner (TD) instances with active MCP servers. Returns port, project name, PID, and instanceId for each instance. Call this at the start of every conversation to discover available instances and choose which one to work with. instanceId is stable for the lifetime of a TD process and is used as target_instance in all other tool calls.

No parameters (other than optional `target_instance`).

### td_project_quit

Save and/or close the current TouchDesigner (TD) project. Can save before closing. Reports if project has unsaved changes. To close a different instance, pass target_instance=instanceId. WARNING: this will shut down the MCP server on that instance.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `save` | boolean | no | Save the project before closing. Default true. |
| `force` | boolean | no | Force close without save dialog. Default false. |

### td_reinit_extension

Reinitialize an extension on a COMP in TouchDesigner (TD). Call this AFTER finishing all code edits via td_write_dat to apply changes. Do NOT call after every small edit - batch your changes first.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Path to the COMP with the extension |

### td_dev_log

Read the last N entries from the MCP dev log. Only available when Devmode is enabled. Shows request/response history.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `count` | integer | no | Number of recent log entries to return |

### td_clear_dev_log

Clear the current MCP dev log by closing the old file and starting a fresh one. Only available when Devmode is enabled.

No parameters (other than optional `target_instance`).

### td_test_session

Manage test sessions, bug reports, and conversation export. IMPORTANT: Do NOT proactively suggest exporting chat or submitting reports. These are tools for specific situations: - export_chat / submit_report: ONLY when the user encounters a BUG with the plugin or TouchDesigner and wants to report it, or when the user explicitly asks to export the conversation. Never suggest this at session end or as routine action. USER PHRASES → ACTIONS: 'разбор тестовых сессий' / 'analyze test sessions' → list, then pull, read meta.json → index.jsonl → calls/. 'разбор репортов' / 'analyze user reports' → list with session='user', then pull by name. 'экспортируй чат' / 'export chat' → (1) export_chat_id → marker, (2) export_chat with session=marker. 'сообщи о проблеме' / 'report bug' → export chat, review for privacy, then submit_report with summary + tags + result_op=file_path. ACTIONS: export_chat_id | export_chat | submit_report | start | note | import_chat | end | list | pull. list: default=auto-detect repo. session='user' for user_reports (dev only). pull: auto-searches both repos. Auto-detects dev vs user Hub access.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `action` | `export_chat_id` / `export_chat` / `submit_report` / `start` / `note` / `import_chat` / `end` / `list` / `pull` | yes | Action: export_chat_id / export_chat / submit_report / start / note / import_chat / end / list / pull |
| `prompt` | string | no | (start) The test prompt/task description |
| `tags` | array | no | (start) Tags for categorization, e.g. ['ui', 'layout'] |
| `text` | string | no | (note) Observation text. (import_chat) Full conversation text. |
| `outcome` | `success` / `partial` / `failure` | no | (end) Result: success / partial / failure |
| `summary` | string | no | (end) Brief summary of what happened |
| `result_op` | string | no | (end) Path to operator to save as result.tox |
| `session` | string | no | (pull) Session name or substring to download |
