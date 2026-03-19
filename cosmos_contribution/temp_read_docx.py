
import zipfile
import xml.etree.ElementTree as ET
import sys

def get_docx_text(path):
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            texts = []
            for paragraph in tree.findall('.//w:p', ns):
                para_text = []
                for run in paragraph.findall('.//w:t', ns):
                    if run.text:
                        para_text.append(run.text)
                if para_text:
                    texts.append("".join(para_text))
            return "\n".join(texts)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    path = r"d:\Cosmos\CosmoSynapse_Hermes_Quantum_Upgrade_Prompt (1).docx"
    text = get_docx_text(path)
    # Force utf-8 output
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)
