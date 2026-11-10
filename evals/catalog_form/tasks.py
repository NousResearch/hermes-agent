"""Battery, stub parameter schemas and canned results for the deferred-catalog form A/B/C.

Task kinds: ``deferred`` (needs one specific deferred tool the prompt does not name; several are
look-alike traps, e.g. the blender ``_for_cli`` twins), ``control`` (eager tools only), ``distractor``
(domain-adjacent, no deferred tool should be used) and ``continuation`` (under-specified; the agent
is expected to ask, the harness answers once like a user would).
"""
import os
import random

B, M, N = "mcp__blender__", "mcp__meshy__", "mcp__nous_support__"
S = {"type": "string"}
I = {"type": "integer"}


def _obj(props=None, required=()):
    return {"type": "object", "properties": props or {}, "required": list(required)}


_MODEL_REF = {"task_id": {**S, "description": "Meshy task id of the model."},
              "model_url": {**S, "description": "URL of a model file."},
              "file_path": {**S, "description": "Local model file."}}
_PARAMS = {
    B + "execute_blender_code": _obj({"code": S}, ["code"]),
    B + "execute_blender_code_for_cli": _obj({"code": S, "blend_file": S}, ["code"]),
    B + "get_object_detail_summary": _obj({"name": S}, ["name"]),
    B + "get_prompt": _obj({"name": S}, ["name"]),
    B + "read_resource": _obj({"uri": S}, ["uri"]),
    B + "get_python_api_docs": _obj({"identifier": S}, ["identifier"]),
    B + "get_screenshot_of_area_as_image": _obj({"area_type": S}),
    B + "jump_to_tab_by_name": _obj({"name": S}, ["name"]),
    B + "jump_to_tab_by_space_type": _obj({"space_type": S}, ["space_type"]),
    B + "jump_to_view3d_object_by_name": _obj({"name": S}, ["name"]),
    B + "jump_to_view3d_object_data_by_name": _obj({"name": S}, ["name"]),
    B + "render_thumbnail_to_path": _obj({"output_path": S}, ["output_path"]),
    B + "render_viewport_to_path": _obj({"output_path": S}, ["output_path"]),
    B + "search_api_docs": _obj({"query": S}, ["query"]),
    B + "search_manual_docs": _obj({"query": S}, ["query"]),
    M + "meshy_animate": _obj({"rig_task_id": S, "action_id": I}, ["rig_task_id", "action_id"]),
    M + "meshy_cancel_task": _obj({"task_id": S}, ["task_id"]),
    M + "meshy_check_balance": _obj(),
    M + "meshy_convert": _obj({**_MODEL_REF, "formats": {"type": "array", "items": S}}, ["formats"]),
    M + "meshy_creative_lab": _obj({"file_path": S, "image_url": S}),
    M + "meshy_download_model": _obj({"task_id": S}, ["task_id"]),
    M + "meshy_get_task_status": _obj({"task_id": S, "wait": {"type": "boolean"}}, ["task_id"]),
    M + "meshy_image_to_3d": _obj({"image_url": S, "file_path": S}),
    M + "meshy_image_to_image": _obj({"prompt": S, "image_url": S}, ["prompt"]),
    M + "meshy_list_animations": _obj({"query": S}),
    M + "meshy_list_models": _obj({"limit": I}),
    M + "meshy_list_tasks": _obj({"limit": I}),
    M + "meshy_list_usage": _obj({"limit": I}),
    M + "meshy_multi_image_to_3d": _obj({"image_urls": {"type": "array", "items": S}}, ["image_urls"]),
    M + "meshy_remesh": _obj({**_MODEL_REF, "target_polycount": I,
                              "topology": {**S, "enum": ["triangle", "quad"]}}),
    M + "meshy_resize": _obj({**_MODEL_REF, "height_mm": {"type": "number"}}),
    M + "meshy_retexture": _obj({**_MODEL_REF, "text_style_prompt": S}),
    M + "meshy_send_to_slicer": _obj(),
    M + "meshy_text_to_3d": _obj({"prompt": S, "art_style": S}, ["prompt"]),
    M + "meshy_text_to_3d_refine": _obj({"preview_task_id": S}, ["preview_task_id"]),
    M + "meshy_text_to_image": _obj({"prompt": S}, ["prompt"]),
    M + "meshy_text_to_motion": _obj({"prompt": S}, ["prompt"]),
    "elevenlabs_music_compose": _obj({"prompt": S, "duration_seconds": I}, ["prompt"]),
    "elevenlabs_music_compose_plan": _obj({"plan": {"type": "object"}}, ["plan"]),
    "elevenlabs_music_draft_plan": _obj({"prompt": S, "duration_seconds": I}, ["prompt"]),
    "elevenlabs_music_finetunes": _obj(),
    "elevenlabs_music_from_video": _obj({"video_paths": {"type": "array", "items": S}}, ["video_paths"]),
    "elevenlabs_music_job": _obj({"job_id": S}, ["job_id"]),
    "elevenlabs_music_stems": _obj({"file_path": S, "mode": {**S, "enum": ["two", "six"]}}, ["file_path"]),
    "elevenlabs_music_upload": _obj({"file_path": S}, ["file_path"]),
    "elevenlabs_sound_effect": _obj({"prompt": S, "duration_seconds": {"type": "number"}, "output_path": S}, ["prompt"]),
    N + "report": _obj({"bundle_id": S, "limit": I}),
    N + "threads": _obj({"thread_id": S, "query": S, "limit": I}),
}


def params_for(name):
    if name in _PARAMS:
        return _PARAMS[name]
    if name.startswith(B) and name.endswith("_for_cli"):
        return _obj({"blend_file": {**S, "description": "Path of the .blend file to open in background Blender."}},
                    ["blend_file"])
    if name.startswith(M):
        return _obj(dict(_MODEL_REF))
    return _obj()


# Read-only lookups a careful agent may make on the way; never counted as a wrong tool.
NEUTRAL = {B + "get_blendfile_summary_path_info", B + "get_blendfile_summary_path_info_for_cli",
           M + "meshy_get_task_status", M + "meshy_list_models", M + "meshy_list_tasks",
           "elevenlabs_music_job"}

_OPEN_FILE = "/home/tek/projects/robot_arm/robot_arm_v7.blend"  # what the connected Blender has open
_NOT_CONNECTED = {"error": "No Blender instance is connected (the add-on socket did not answer). "
                           "Tools without the _for_cli suffix need a running Blender."}
_MISSING = {"missing": [{"type": "image", "path": "//textures/rust_normal_4k.png"},
                        {"type": "image", "path": "//hdri/studio_small_08.exr"}], "total_missing": 2}
_LINKED = {"direct": ["//lib/char_mira_rig.blend", "//lib/props_market.blend"],
           "indirect": [{"path": "//lib/mat_fabric.blend", "via": "//lib/props_market.blend"}]}


def _touch(path):
    """Leave the output file a real tool would have written, so the agent's own check passes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"ID3\x04\x00\x00\x00\x00\x00\x23" + random.Random(len(path)).randbytes(48_000))


def canned_result(name, args, task_id, ws):
    """Realistic result for one stub call. Live blender tools answer for the open file (or fail when
    the task says Blender is not running); ``_for_cli`` tools answer for the file they were given."""
    blender_running = task_id not in ("bl_cli_datablocks",)
    if name.startswith(B) and not name.endswith("_for_cli") and not blender_running and name.split("__")[-1] not in (
            "get_python_api_docs", "search_api_docs", "search_manual_docs", "list_prompts", "list_resources"):
        return _NOT_CONNECTED
    short = name.split("__")[-1]
    blend = str(args.get("blend_file") or "")
    if short == "get_blendfile_summary_datablocks_for_cli":
        if blend.endswith("castle_keep.blend"):
            return {"blend_file": blend, "datablocks": {"meshes": 47, "materials": 13, "images": 21, "objects": 62,
                                                        "collections": 6, "cameras": 2, "lights": 5}}
        return {"error": f"cannot open {blend!r}"}
    if short == "get_blendfile_summary_datablocks":
        return {"filepath": _OPEN_FILE, "datablocks": {"meshes": 9, "materials": 4, "images": 3, "objects": 14},
                "active_workspace": "Layout"}
    if short == "get_blendfile_summary_missing_files":
        return {"filepath": _OPEN_FILE, **_MISSING}
    if short == "get_blendfile_summary_missing_files_for_cli":
        return {"blend_file": blend, **_MISSING} if blend == _OPEN_FILE else {"blend_file": blend, "missing": [], "total_missing": 0}
    if short == "get_blendfile_summary_of_linked_libraries_for_cli":
        return {"blend_file": blend, **_LINKED} if blend.endswith("shot_042.blend") else {"error": f"cannot open {blend!r}"}
    if short == "get_blendfile_summary_of_linked_libraries":
        return {"filepath": _OPEN_FILE, "direct": ["//lib/servo_parts.blend"], "indirect": []}
    if short in ("get_blendfile_summary_path_info", "get_blendfile_summary_path_info_for_cli"):
        return {"filepath": blend or _OPEN_FILE, "is_saved": True, "is_dirty": short.endswith("path_info"),
                "blender_version": "5.1.0"}
    if short == "get_object_detail_summary":
        if str(args.get("name")) == "Gear_Large":
            return {"name": "Gear_Large", "type": "MESH", "vertices": 2304, "location": [0.0, 0.0, 0.42],
                    "modifiers": [{"name": "Bevel", "type": "BEVEL"}, {"name": "Solidify", "type": "SOLIDIFY"},
                                  {"name": "Array", "type": "ARRAY", "count": 6}], "materials": ["Brass_Worn"]}
        return {"error": f"no object named {args.get('name')!r}"}
    if short == "get_objects_summary":
        return {"collections": {"Collection": ["Base", "Gear_Large", "Gear_Small", "Arm_Upper", "Arm_Lower", "Camera", "Sun"]}}
    if short in ("execute_blender_code", "execute_blender_code_for_cli"):
        return {"status": "ok", "stdout": "", "note": "code ran; nothing was printed"}
    if short == "meshy_analyze_printability":
        return {"task_id": args.get("task_id"), "printable": False, "watertight": False, "non_manifold_edges": 14,
                "holes": 3, "min_wall_thickness_mm": 0.4, "recommendation": "run a printability repair before slicing"}
    if short == "meshy_repair_printability":
        return {"task_id": "019b-repair-8802", "status": "PENDING", "credits_charged": 10}
    if short == "meshy_check_balance":
        return {"balance": 3460, "unit": "credits"}
    if short == "meshy_list_usage":
        return {"usage": [{"task_id": "019a-77fe-knight", "type": "text-to-3d", "credits": 20, "finished_at": "2026-11-07"},
                          {"task_id": "019a-6c11-fox", "type": "remesh", "credits": 5, "finished_at": "2026-11-05"}]}
    if short == "meshy_remesh":
        return {"task_id": "019b-remesh-5521", "status": "PENDING", "target_polycount": args.get("target_polycount")}
    if short == "meshy_get_task_status":
        return {"task_id": args.get("task_id"), "status": "SUCCEEDED", "progress": 100}
    if short == "meshy_text_to_3d":
        return {"task_id": "019b-preview-3307", "status": "PENDING", "mode": "preview", "prompt": args.get("prompt")}
    if short == "meshy_text_to_image":
        return {"task_id": "019b-image-1140", "status": "PENDING"}
    if short == "elevenlabs_music_stems":
        stem = os.path.splitext(os.path.basename(str(args.get("file_path") or "song")))[0]
        paths = {"vocals": os.path.join(ws, "audio", f"{stem}.vocals.mp3"),
                 "accompaniment": os.path.join(ws, "audio", f"{stem}.accompaniment.mp3")}
        for path in paths.values():
            _touch(path)
        return {"success": True, "stems": paths}
    if short == "elevenlabs_music_draft_plan":
        return {"success": True, "credits_charged": 0, "plan": {"positive_global_styles": ["lo-fi", "mellow", "vinyl crackle"],
                "sections": [{"section_name": "Intro", "duration_ms": 8000}, {"section_name": "Groove", "duration_ms": 16000},
                             {"section_name": "Outro", "duration_ms": 6000}]}, "plan_id": "plan_lofi_7741"}
    if short in ("elevenlabs_music_compose", "elevenlabs_music_compose_plan"):
        return {"success": True, "job_id": "job_music_2290", "status": "running", "credits_charged": 300}
    if short == "elevenlabs_sound_effect":
        _touch(os.path.join(ws, "sfx", "door_creak_01.mp3"))
        return {"success": True, "path": os.path.join(ws, "sfx", "door_creak_01.mp3"), "duration_seconds": args.get("duration_seconds") or 3}
    if short == "threads":
        return {"threads": [{"id": "t-9913", "title": "Telegram gateway drops voice notes after update", "replies": 7},
                            {"id": "t-9908", "title": "Skills hub install fails behind corporate proxy", "replies": 3},
                            {"id": "t-9897", "title": "Cron job runs twice after laptop wakes", "replies": 12}]}
    if short == "report":
        return {"bundles": [{"id": "dbg-5a1f", "uploaded": "2026-11-08", "hermes_version": "0.31.2"}]}
    return {"success": True, "result": "ok", "items": []}


def _poly(a):
    try:
        return 4000 <= int(a.get("target_polycount")) <= 6000 and "019a-77fe-knight" in str(a.get("task_id"))
    except (TypeError, ValueError):
        return False


def _file_has(rel, *needles):
    def check(ws):
        try:
            with open(os.path.join(ws, rel), encoding="utf-8") as fh:
                text = fh.read().lower()
        except OSError:
            return False
        return all(n.lower() in text for n in needles)
    return check


TASKS = {
    # ---- deferred: blender (look-alike _for_cli twins)
    "bl_cli_datablocks": {
        "kind": "deferred", "targets": [B + "get_blendfile_summary_datablocks_for_cli"], "facts": ["47", "13"],
        "prompt": "There's a Blender file at {ws}/assets/castle_keep.blend. Blender isn't running right now and I "
                  "don't want a window opened. How many meshes and how many materials are in that file?"},
    "bl_live_missing": {
        "kind": "deferred", "targets": [B + "get_blendfile_summary_missing_files"], "facts": ["rust_normal_4k", "studio_small_08"],
        "prompt": "In the Blender scene I have open right now, are any textures or other external files missing from disk? "
                  "List them."},
    "bl_cli_linked": {
        "kind": "deferred", "targets": [B + "get_blendfile_summary_of_linked_libraries_for_cli"],
        "facts": ["char_mira_rig", "props_market", "mat_fabric"],
        "prompt": "Which library .blend files does {ws}/shots/shot_042.blend pull in, directly or indirectly? Leave my "
                  "running Blender session alone, I'm in the middle of a sculpt in a different file."},
    "bl_object_modifiers": {
        "kind": "deferred", "targets": [B + "get_object_detail_summary"], "facts": ["bevel", "solidify", "array"],
        "neutral": [B + "get_objects_summary", B + "execute_blender_code"],
        "prompt": "In my open Blender scene, which modifiers are on the object named Gear_Large?"},
    # ---- deferred: meshy
    "meshy_printability": {
        "kind": "deferred", "targets": [M + "meshy_analyze_printability"], "facts": ["14", ["watertight", "water-tight"]],
        "prompt": "My Meshy model with task id 019a-77fe-knight: could I 3D print it as it is? Tell me what's wrong "
                  "with it if anything. Don't change the model."},
    "meshy_balance": {
        "kind": "deferred", "targets": [M + "meshy_check_balance"], "facts": [["3460", "3,460"]],
        "prompt": "How many Meshy credits do I have left?"},
    "meshy_remesh": {
        "kind": "deferred", "targets": [M + "meshy_remesh"], "arg_ok": _poly, "facts": ["019b-remesh-5521"],
        "prompt": "Bring my Meshy model 019a-77fe-knight down to about 5,000 triangles and give me the id of the job you started."},
    # ---- deferred: elevenlabs_music (plugin server)
    "el_stems": {
        "kind": "deferred", "targets": ["elevenlabs_music_stems"], "facts": ["demo_take3.vocals.mp3"],
        "prompt": "Split {ws}/audio/demo_take3.mp3 into a vocals file and a backing-track file and tell me where they are."},
    "el_draft_plan": {
        "kind": "deferred", "targets": ["elevenlabs_music_draft_plan"], "facts": ["intro", "outro"],
        "prompt": "I want a 30 second lo-fi study beat eventually, but don't spend any credits yet. Just get me the "
                  "section-by-section outline the music service would use for it."},
    # ---- deferred: nous-support
    "ns_threads": {
        "kind": "deferred", "targets": [N + "threads"], "facts": ["voice notes", "proxy", "twice"],
        "prompt": "What are people asking about in our support Discord channel lately? Give me the three most recent thread titles."},
    # ---- deferred built-in plugin tool (full line in every arm)
    "core_sfx": {
        "kind": "deferred", "targets": ["elevenlabs_sound_effect"], "facts": ["door_creak_01.mp3"],
        "prompt": "Make me a 3 second creaky door sound effect and tell me the file path."},
    # ---- continuation: under-specified, answered once by the scripted user
    "meshy_underspecified": {
        "kind": "continuation", "targets": [M + "meshy_text_to_3d"], "facts": ["019b-preview-3307"],
        "replies": ["A low-poly fox. Just the untextured preview is fine, give me the task id."],
        "prompt": "Use Meshy to make me a 3D model."},
    # ---- controls: eager tools only
    "ctl_count_refunds": {
        "kind": "control", "facts": ["7"],
        "prompt": "How many rows in {ws}/data/orders.csv have the status refunded? Answer with the number."},
    "ctl_write_todo": {
        "kind": "control", "file_check": _file_has("notes/todo.md", "alpha", "beta", "gamma"),
        "prompt": "Create {ws}/notes/todo.md with three bullet items: alpha, beta, gamma."},
    # ---- distractors: domain-adjacent, no deferred tool should be used
    "dis_models_csv": {
        "kind": "distractor", "facts": ["dragon_statue"],
        "prompt": "{ws}/data/models.csv lists my 3D models with their triangle counts. Which model has the most triangles?"},
    "dis_blender_script": {
        "kind": "distractor", "file_check": _file_has("scripts/add_cube.py", "bpy", "cube"),
        "prompt": "Write a Blender Python script at {ws}/scripts/add_cube.py that adds a cube at the origin. Just write "
                  "the file, I'll run it myself later."},
}
TASK_ORDER = list(TASKS)


def write_fixtures(ws):
    for d in ("assets", "shots", "audio", "data", "notes", "scripts", "sfx"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    # Opaque binaries, like real ones: a zstd-compressed .blend and an mp3 are not readable from a shell.
    for seed, rel, magic, size in ((11, "assets/castle_keep.blend", b"\x28\xb5\x2f\xfd", 240_000),
                                   (12, "shots/shot_042.blend", b"\x28\xb5\x2f\xfd", 310_000),
                                   (13, "audio/demo_take3.mp3", b"ID3\x04\x00\x00\x00\x00\x00\x23", 180_000)):
        with open(os.path.join(ws, rel), "wb") as fh:
            fh.write(magic + random.Random(seed).randbytes(size))
    statuses = ["paid", "refunded", "paid", "shipped", "refunded", "paid", "cancelled", "refunded", "paid", "paid",
                "refunded", "shipped", "paid", "refunded", "paid", "refunded", "shipped", "paid", "refunded", "paid",
                "paid", "unrefunded-hold"]
    with open(os.path.join(ws, "data", "orders.csv"), "w", encoding="utf-8") as fh:
        fh.write("order_id,amount,status\n")
        fh.writelines(f"{1000 + i},{19 + i * 3}.00,{st}\n" for i, st in enumerate(statuses))
    with open(os.path.join(ws, "data", "models.csv"), "w", encoding="utf-8") as fh:
        fh.write("model,triangles,format\nknight,48210,glb\nfox_lowpoly,1920,glb\ndragon_statue,912044,stl\n"
                 "market_stall,90118,fbx\ngear_large,2304,obj\n")
