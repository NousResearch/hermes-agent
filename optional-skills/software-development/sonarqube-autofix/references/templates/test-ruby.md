# Ruby Test Template (RSpec)

```ruby
require 'rspec'
require 'example'

RSpec.describe ExampleClass do
  let(:mock_dependency) { double('Dependency') }
  let(:example) { ExampleClass.new(mock_dependency) }

  describe '#method' do
    it 'should return expected value' do
      # Given
      allow(mock_dependency).to receive(:get_value).and_return('expected')

      # When
      result = example.method

      # Then
      expect(result).to eq('expected')
      expect(mock_dependency).to have_received(:get_value).once
    end
  end
end
```

## Patterns

- Use RSpec
- Use double for mocks
- Given-When-Then pattern
- Verify all mock calls
- Use let for lazy setup
