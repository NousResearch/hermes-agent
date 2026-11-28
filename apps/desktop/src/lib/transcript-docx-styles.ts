const W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

export const DOCX_STYLES = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="${W}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos" w:cs="Arial"/><w:color w:val="253247"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="288" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:next w:val="Metadata"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="0" w:after="140"/></w:pPr><w:rPr><w:b/><w:color w:val="16283B"/><w:sz w:val="40"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Metadata"><w:name w:val="Transcript metadata"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="100"/></w:pPr><w:rPr><w:color w:val="596779"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="300" w:after="100"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="176B74"/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Heading2"/><w:pPr><w:ind w:left="360"/><w:outlineLvl w:val="2"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="TranscriptCode"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="60"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="596779"/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TranscriptCode"><w:name w:val="Transcript code"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/><w:shd w:val="clear" w:fill="F2F5F8"/><w:ind w:left="120" w:right="120"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TranscriptReasoning"><w:name w:val="Transcript reasoning"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="240" w:right="120"/><w:spacing w:after="100"/></w:pPr><w:rPr><w:color w:val="526078"/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Warning"><w:name w:val="Transcript note"/><w:basedOn w:val="Metadata"/><w:rPr><w:i/><w:color w:val="64523B"/></w:rPr></w:style>
<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/><w:rPr><w:color w:val="176B74"/><w:u w:val="single"/></w:rPr></w:style>
</w:styles>`

export const DOCX_SETTINGS = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="${W}"><w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat><w:updateFields w:val="true"/></w:settings>`

export const DOCX_FOOTER = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr xmlns:w="${W}"><w:p><w:pPr><w:pStyle w:val="Metadata"/></w:pPr><w:r><w:t xml:space="preserve">Hermes conversation transcript · Page </w:t></w:r><w:fldSimple w:instr="PAGE"><w:r><w:t>1</w:t></w:r></w:fldSimple></w:p></w:ftr>`
