
import docx
import sys

def read_docx(file_path):
    doc = docx.Document(file_path)
    fullText = []
    
    # Iterate through all story elements (paragraphs and tables)
    for element in doc.element.body:
        if element.tag.endswith('p'):
            para = docx.text.paragraph.Paragraph(element, doc)
            if para.text.strip():
                fullText.append(para.text)
        elif element.tag.endswith('tbl'):
            table = docx.table.Table(element, doc)
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    row_text.append(cell.text.strip())
                fullText.append(" | ".join(row_text))
    
    return '\n'.join(fullText)

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    try:
        print(read_docx(sys.argv[1]))
    except Exception as e:
        print(f"Error: {e}")
