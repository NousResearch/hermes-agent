import subprocess, sys, os

try:
    import PyPDF2
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyPDF2', '-q'])
    import PyPDF2

folder = r'd:\Cosmos\old docs'
outfile = r'd:\Cosmos\pdf_output.txt'

with open(outfile, 'w', encoding='utf-8', errors='replace') as out:
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.pdf'):
            continue
        out.write("=" * 80 + "\n")
        out.write("FILE: " + fname + "\n")
        out.write("=" * 80 + "\n\n")
        path = os.path.join(folder, fname)
        reader = PyPDF2.PdfReader(path)
        out.write(f"Pages: {len(reader.pages)}\n\n")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                out.write(f"--- Page {i+1} ---\n")
                out.write(text + "\n\n")

print(f"Wrote output to {outfile}")
print(f"Size: {os.path.getsize(outfile)} bytes")
