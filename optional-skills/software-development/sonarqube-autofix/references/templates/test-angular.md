# Angular Test Template (Karma + Jasmine)

Template for unit tests in Angular using Karma and TestBed.

## Basic Structure

```typescript
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { {{ComponentName}} } from './{{component-file}}';

describe('{{ComponentName}}', () => {
  let component: {{ComponentName}};
  let fixture: ComponentFixture<{{ComponentName}};
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ {{ComponentName}} ],
      imports: [HttpClientTestingModule],
      providers: [
        // Add any necessary providers here
      ]
    }).compileComponents();

    fixture = TestBed.createComponent({{ComponentName}});
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  // Add specific tests here
});
```

## Component Tests with Inputs

```typescript
describe('{{ComponentName}}', () => {
  let component: {{ComponentName}};
  let fixture: ComponentFixture<{{ComponentName}};

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ {{ComponentName}} ]
    }).compileComponents();

    fixture = TestBed.createComponent({{ComponentName}});
    component = fixture.componentInstance;
  });

  it('should display input value', () => {
    component.{{inputName}} = '{{testValue}}';
    fixture.detectChanges();
    const element = fixture.nativeElement.querySelector('.{{css-class}}');
    expect(element.textContent).toContain('{{testValue}}');
  });
});
```

## Service Tests

```typescript
import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { {{ServiceName}} } from './{{service-file}}';

describe('{{ServiceName}}', () => {
  let service: {{ServiceName}};
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [ {{ServiceName}} ]
    });
    service = TestBed.inject({{ServiceName}});
    httpMock = TestBed.inject(HttpTestingController);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should fetch data from API', () => {
    const mockData = { id: 1, name: 'Test' };
    service.getData().subscribe(data => {
      expect(data).toEqual(mockData);
    });

    const req = httpMock.expectOne('/api/data');
    expect(req.request.method).toBe('GET');
    req.flush(mockData);
  });
});
```

## Pipe Tests

```typescript
import { {{PipeName}} } from './{{pipe-file}}';

describe('{{PipeName}}', () => {
  it('should transform value', () => {
    const pipe = new {{PipeName}}();
    expect(pipe.transform('{{input}}')).toBe('{{expectedOutput}}');
  });
});
```

## Directive Tests

```typescript
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { {{DirectiveName}} } from './{{directive-file}}';

describe('{{DirectiveName}}', () => {
  let component: TestComponent;
  let fixture: ComponentFixture<TestComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ {{DirectiveName}}, TestComponent ]
    }).compileComponents();

    fixture = TestBed.createComponent(TestComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should apply directive behavior', () => {
    const element = fixture.nativeElement.querySelector('.{{css-class}}');
    expect(element).toHaveClass('{{expected-class}}');
  });
});

@Component({
  template: '<div {{directiveName}} class="{{css-class}}">Test</div>'
})
class TestComponent {}
```

## Best Practices

- Use `TestBed` to set up the test environment
- Use `HttpClientTestingModule` for HTTP tests
- Use `async/await` for async setup
- Call `fixture.detectChanges()` after changing properties
- Use `httpMock.verify()` to ensure all requests were handled
- Test behavior, not implementation
- Use clear, descriptive test names (Given/When/Then)
