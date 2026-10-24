import { strToU8, zipSync } from 'fflate'
import { describe, expect, it } from 'vitest'

import { docxPreviewDocument } from './office-document'

function docxFixture(): Uint8Array {
  const contentTypes = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>`

  const rels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>`

  const document = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hermes Word probe</w:t></w:r></w:p></w:body></w:document>`

  return zipSync({
    '[Content_Types].xml': strToU8(contentTypes),
    '_rels/.rels': strToU8(rels),
    'word/document.xml': strToU8(document)
  })
}

describe('Word preview', () => {
  it('renders docx text into a sandboxed document', async () => {
    const html = await docxPreviewDocument(docxFixture())

    expect(html).toContain('Hermes Word probe')
    expect(html).toContain('class="page"')
    expect(html).toContain('background:#e7e6e6')
    expect(html).toContain("default-src 'none'")
    expect(html).not.toContain('<script')
  })
})
