export { PluginPage } from "./PluginPage";
export { exposePluginSDK, getPluginComponent, getRegisteredCount, onPluginRegistered } from "./registry";
export { getSlotEntries, KNOWN_SLOT_NAMES, onSlotRegistered, PluginSlot, registerSlot, unregisterPluginSlots } from "./slots";
export type { KnownSlotName } from "./slots";
export type { PluginManifest, RegisteredPlugin } from "./types";
export { usePlugins } from "./usePlugins";
