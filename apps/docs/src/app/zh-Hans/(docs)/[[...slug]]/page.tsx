import { DocumentPage, documentMetadata, documentParams } from '@/components/document-page';

interface Props { params: Promise<{ slug?: string[] }> }
export const dynamicParams = false;
export const generateStaticParams = () => documentParams('zh-Hans');
export async function generateMetadata({ params }: Props) {
  return documentMetadata((await params).slug, 'zh-Hans');
}
export default async function Page({ params }: Props) {
  return <DocumentPage slugs={(await params).slug} locale="zh-Hans" />;
}
