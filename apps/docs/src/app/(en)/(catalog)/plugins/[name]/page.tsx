// English per-plugin page (/plugins/<name>), generated at build time from the same
// public/api/plugins.json snapshot the catalog grid fetches — the replacement for the
// Docusaurus plugin-catalog-pages addRoute step.
import { notFound } from "next/navigation";
import PluginPage from "@/components/PluginCatalog/PluginPage";
import { decodeCatalogParam, getPluginPageData, listPluginPageParams } from "@/lib/plugin-pages";
import { pluginPageMetadata } from "@/lib/catalog-routes";

export async function generateMetadata(props: { params: Promise<{ name: string }> }) {
  return pluginPageMetadata(props, "en");
}

export async function generateStaticParams(): Promise<Array<{ name: string }>> {
  return (await listPluginPageParams()).map((name) => ({ name }));
}

export default async function Page({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  const data = await getPluginPageData(decodeCatalogParam(name));
  if (!data) notFound();
  return <PluginPage data={data} locale="en" />;
}
