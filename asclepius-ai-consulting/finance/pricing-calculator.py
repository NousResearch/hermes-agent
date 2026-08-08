#!/usr/bin/env python3
import argparse
RANGES={'diagnostic':(999,999),'build':(2500,10000),'retainer':(500,2000),'office-hours':(150,300)}
def calculate(tier,hours,complexity):
    lo,hi=RANGES[tier]; base=hours*150*complexity
    price=max(lo,min(hi,base)); value=price/max(hours,0.1); return round(price,2), round(value,2)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--tier',choices=RANGES,required=True); p.add_argument('--hours',type=float,required=True); p.add_argument('--complexity',type=float,default=1.0); a=p.parse_args(); price,value=calculate(a.tier,a.hours,a.complexity); print(f'Suggested price: ${price:.2f}'); print(f'Blended value per Richie hour: ${value:.2f}/hour')
if __name__=='__main__': main()
