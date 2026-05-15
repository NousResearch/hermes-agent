from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnostics import diagnose_path
from .image_modes import render_image
from .presets import PRESETS
from .preview import render_preview
from .video_modes import build_eikon_from_frames, build_eikon_from_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='ascii-pipeline')
    sub = parser.add_subparsers(dest='cmd')

    presets = sub.add_parser('presets', help='List canonical presets')
    presets.add_argument('--verbose', action='store_true')

    diagnose = sub.add_parser('diagnose', help='Analyze a text or .eikon file')
    diagnose.add_argument('--input', required=True)
    diagnose.add_argument('--expected-width', type=int, default=48)
    diagnose.add_argument('--expected-height', type=int, default=24)
    diagnose.add_argument('--pretty', action='store_true', help='Indent JSON output')

    preview = sub.add_parser('render-preview', help='Render a text or .eikon frame to PNG')
    preview.add_argument('--input', required=True)
    preview.add_argument('--out', required=True)
    preview.add_argument('--frame', type=int, default=0)
    preview.add_argument('--font-size', type=int, default=18)

    render_image_parser = sub.add_parser('render-image', help='Render a still image to ASCII text using a named preset')
    render_image_parser.add_argument('--input', required=True)
    render_image_parser.add_argument('--out', required=True)
    render_image_parser.add_argument('--preset', default='stroke-clarity', choices=sorted(PRESETS.keys()))
    render_image_parser.add_argument('--preview-out', default=None)
    render_image_parser.add_argument('--diagnostics-out', default=None)
    render_image_parser.add_argument('--fullsize', action='store_true', help='Emit showcase-tier 192x96 output (equivalent to --scale 4)')
    render_image_parser.add_argument('--scale', type=int, default=None, help='Integer multiplier of base grid (48x24 × N). Max 16. Use --fullsize for equivalent of --scale 4.')
    render_image_parser.add_argument('--pretty', action='store_true', help='Indent JSON output')

    build = sub.add_parser('build-eikon', help='Build a full-size eikon from extracted PNG frames')
    build.add_argument('--frames-dir', required=True)
    build.add_argument('--out', required=True)
    build.add_argument('--grid', default='192x96')
    build.add_argument('--charset', default='dense-ref')
    build.add_argument('--collapse-to', default=None)
    build.add_argument('--id', default=None)
    build.add_argument('--pretty', action='store_true')

    build_video = sub.add_parser('build-eikon-from-video', help='Build an animated eikon directly from a video file')
    build_video.add_argument('--input', required=True, help='Path to source video (MP4, MOV, etc.)')
    build_video.add_argument('--out', required=True, help='Destination .eikon file')
    build_video.add_argument('--grid', default='192x96', help='ASCII grid size WxH (default 192x96)')
    build_video.add_argument('--charset', default='dense-ref', choices=['dense-ref', 'd30'])
    build_video.add_argument('--fps', type=float, default=None, help='Extract frames at this rate (default: native fps)')
    build_video.add_argument('--no-motion-phases', dest='motion_phases', action='store_false', help='Disable auto state assignment; all frames become idle')
    build_video.add_argument('--states', nargs=3, default=['idle', 'thinking', 'speaking'], metavar=('IDLE', 'THINKING', 'SPEAKING'), help='Three state names for motion clustering')
    build_video.add_argument('--id', default=None, help='Eikon identifier (default: output filename stem)')
    build_video.add_argument('--pretty', action='store_true', help='Indent JSON output')

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == 'presets':
        for name, preset in PRESETS.items():
            if args.verbose:
                print(
                    f'{name}: backend={preset.backend} target={preset.target} '
                    f'source_scale={preset.source_scale} :: {preset.notes}'
                )
            else:
                print(name)
        return

    if args.cmd == 'diagnose':
        summary = diagnose_path(
            args.input,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
        )
        print(json.dumps(summary, indent=2 if args.pretty else None))
        return

    if args.cmd == 'render-preview':
        output = render_preview(
            args.input,
            args.out,
            frame_index=args.frame,
            font_size=args.font_size,
        )
        print(str(Path(output)))
        return

    if args.cmd == 'render-image':
        # Resolve scale: --fullsize implies --scale 4; mutual exclusivity checked earlier or implicitly handled
        if args.fullsize and args.scale is not None:
            parser.error("--fullsize and --scale cannot be used together")
        scale = args.scale if args.scale is not None else (4 if args.fullsize else 1)

        result = render_image(
            args.input,
            args.out,
            preset_name=args.preset,
            preview_out=args.preview_out,
            diagnostics_out=args.diagnostics_out,
            scale=scale,
        )
        print(json.dumps(result, indent=2 if args.pretty else None))
        return

    if args.cmd == 'build-eikon':
        result = build_eikon_from_frames(
            args.frames_dir,
            args.out,
            grid=args.grid,
            charset=args.charset,
            collapse_to=args.collapse_to,
            eikon_id=args.id,
        )
        print(json.dumps(result, indent=2 if args.pretty else None))
        return

    if args.cmd == 'build-eikon-from-video':
        # Normalize states list
        states = args.states
        result = build_eikon_from_video(
            args.input,
            args.out,
            grid=args.grid,
            charset=args.charset,
            fps=args.fps,
            states=states,
            motion_phases=args.motion_phases,
            eikon_id=args.id,
        )
        print(json.dumps(result, indent=2 if args.pretty else None))
        return

    parser.print_help()


if __name__ == '__main__':
    main()
