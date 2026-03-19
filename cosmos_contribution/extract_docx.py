
import zipfile
import xml.etree.ElementTree as ET
import os

def get_docx_text(path):
    """
    Extracts text from a docx file without external libraries.
    """
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # Namespaces are important in Word XML
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

path = r"d:\Cosmos\CosmoSynapse_Swarm_Upgrade_Guide.docx"
text = get_docx_text(path)
import sys
# Force utf-8 output to avoid CP1252 issues on Windows terminal
sys.stdout.reconfigure(encoding='utf-8')
print(text)
