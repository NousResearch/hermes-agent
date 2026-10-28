// zh-Hans author page (/zh-Hans/plugins/by/<slug>), generated at build time from the
// same public/api/plugins.json snapshot as the English pages.
import { notFound } from "next/navigation";
import AuthorPage from "@/components/PluginCatalog/AuthorPage";
import { decodeCatalogParam, getAuthorPageData, listAuthorPageParams } from "@/lib/plugin-pages";
import { authorPageMetadata } from "@/lib/catalog-routes";

export async function generateMetadata(props: { params: Promise<{ author: string }> }) {
  return authorPageMetadata(props, "zh-Hans");
}

export async function generateStaticParams(): Promise<Array<{ author: string }>> {
  return (await listAuthorPageParams()).map((author) => ({ author }));
}

export default async function Page({ params }: { params: Promise<{ author: string }> }) {
  const { author } = await params;
  const data = await getAuthorPageData(decodeCatalogParam(author));
  if (!data) notFound();
  return <AuthorPage data={data} locale="zh-Hans" />;
}
