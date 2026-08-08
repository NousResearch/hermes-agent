import subprocess, sys, tempfile, unittest
from pathlib import Path
CRM=Path(__file__).parents[1]/'crm'/'crm.py'
class CRMTests(unittest.TestCase):
    def run_crm(self,db,*args):
        return subprocess.run([sys.executable,str(CRM),'--db',str(db),*args],text=True,capture_output=True,check=True).stdout
    def test_add_update_revenue(self):
        with tempfile.TemporaryDirectory() as d:
            db=Path(d)/'t.db'
            self.run_crm(db,'add-lead','--name','A','--business','B','--source','warm')
            self.assertIn('A | B', self.run_crm(db,'list-leads'))
            self.run_crm(db,'update-status','--lead-id','1','--status','qualified')
            self.assertIn('qualified', self.run_crm(db,'list-leads'))
            self.run_crm(db,'add-engagement','--lead-id','1','--tier','Diagnostic','--price','999','--status','closed_won')
            out=self.run_crm(db,'revenue-summary')
            self.assertIn('Diagnostic | $999.00', out); self.assertIn('TOTAL | $999.00', out)
if __name__=='__main__': unittest.main()
