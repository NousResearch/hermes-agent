// English Plugin Catalog index (/plugins → /docs/plugins with the app-wide basePath).
import type { Metadata } from "next";
import PluginCatalogGrid from "@/components/PluginsGrid";

export const metadata: Metadata = {
  title: "Plugin Catalog",
  description: "Give Hermes new powers: reviewed plugins you can install in one click",
};

export default function Page() {
  return <PluginCatalogGrid locale="en" />;
}
