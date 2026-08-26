import { jsx as _jsx } from "react/jsx-runtime";
import { $accentOverride, PALETTE_AREA, setAccentOverride, STATUSBAR_AREAS } from '@hermes/plugin-sdk';
import { AccentPickerTrigger } from './picker';
const plugin = {
    id: 'accent',
    name: 'Accent Picker',
    description: 'Pick the theme accent from an OKLCH color picker in the status bar; the palette re-derives live. Authoring tool — the color is not persisted.',
    defaultEnabled: false,
    register(ctx) {
        // The override is a scratch value, not a setting. Dropping it on unregister
        // means disabling the plugin (or reloading) returns every surface to the
        // authored theme instead of stranding a color with no control to clear it.
        ctx.onDispose(() => setAccentOverride(null));
        ctx.registerMany([
            {
                id: 'picker',
                area: STATUSBAR_AREAS.right,
                order: 90,
                render: () => _jsx(AccentPickerTrigger, {})
            },
            {
                id: 'reset',
                area: PALETTE_AREA,
                data: {
                    id: 'accent.reset',
                    label: 'Accent: reset to the theme default',
                    keywords: ['accent', 'color', 'theme', 'reset', 'default'],
                    run: () => setAccentOverride(null)
                }
            },
            {
                id: 'copy',
                area: PALETTE_AREA,
                data: {
                    id: 'accent.copy',
                    label: 'Accent: copy the current color',
                    keywords: ['accent', 'color', 'hex', 'copy', 'clipboard'],
                    run: () => {
                        const hex = $accentOverride.get();
                        if (hex) {
                            void navigator.clipboard?.writeText(hex);
                        }
                    }
                }
            }
        ]);
    }
};
export default plugin;
