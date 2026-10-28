// zh-Hans Plugin Catalog index (/zh-Hans/plugins → /docs/zh-Hans/plugins).
import type { Metadata } from "next";
import PluginCatalogGrid from "@/components/PluginsGrid";

export const metadata: Metadata = {
  title: "Plugin Catalog",
  description: "Give Hermes new powers: reviewed plugins you can install in one click",
};

export default function Page() {
  return <PluginCatalogGrid locale="zh-Hans" />;
}
