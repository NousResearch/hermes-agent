// zh-Hans per-plugin page (/zh-Hans/plugins/<name>), generated at build time from the
// same public/api/plugins.json snapshot as the English pages (catalog data is
// locale-independent; only internal links pick up the zh-Hans prefix).
import { notFound } from "next/navigation";
import PluginPage from "@/components/PluginCatalog/PluginPage";
import { decodeCatalogParam, getPluginPageData, listPluginPageParams } from "@/lib/plugin-pages";
import { pluginPageMetadata } from "@/lib/catalog-routes";

export async function generateMetadata(props: { params: Promise<{ name: string }> }) {
  return pluginPageMetadata(props, "zh-Hans");
}

export async function generateStaticParams(): Promise<Array<{ name: string }>> {
  return (await listPluginPageParams()).map((name) => ({ name }));
}

export default async function Page({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  const data = await getPluginPageData(decodeCatalogParam(name));
  if (!data) notFound();
  return <PluginPage data={data} locale="zh-Hans" />;
}
