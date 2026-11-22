// Registered via `node --import ./scripts/ts-extension-resolver.mjs`.
// See scripts/ts-extension-hooks.mjs for the resolve hook itself.
import { register } from 'node:module';

register(new URL('./ts-extension-hooks.mjs', import.meta.url));
